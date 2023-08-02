import logging
from typing import cast
import numpy as np
from PIL import Image
import itertools
import yaml
from sid_utils.semafor_api import artifact_graph, evidence_graph, asset
from sid_utils.misc.feature_extractors import file_size_extractor
from sid_utils.semafor_api.sources import SOCIAL_MEDIA_SOURCES_EVAL2
from sid_utils.semafor_api.evidence_graph import group_nodes_by_type
from semafor.semafor_component import SemaforComponent
import semafor.util as util
import semafor.data as data
from semafor.data import (
    Component, SemaforMessage, ProbeRequestMessage, ActionTypeSpec,
    EvAnalysisNode, EvReferenceNode, EvAttributionNode, EvEdge,
    ProbeResponseMessage, EvScore, EvNonSemanticConsistencyCheckNode,
    EvSemanticConsistencyCheckNode,
    EvConceptNode, EvIgnoredAssetNode, EvidenceGraph, ArtifactGraph,
    AgImageNode, EvDetectionNode, EvAnalyticModelNode,
    EvSemanticCategory, AnalyticScope, EgAssociatedGraphReference,
    EvImageLocBoundingPolyNode, EvImageLocPixelMapNode, EvCharacterizationNode
)
from semafor.helpers.evigraphs import (append_node, append_edge, opt_out,
                                       opt_out_all, get_associated_graph_id,
                                       create_evidence_graph, find_nodes_by_type,
                                       find_incoming_edges)
from semafor.helpers import evigraphs
import semafor.helpers.evigraphs as ev_helper
from semafor.helpers.artigraphs import find_node
from semafor.helpers.probes import find_analyzed_egs
from semafor._server.models.artifact_graph import ArtifactGraph
from semafor._server.models.evidence_graph import EvidenceGraph
from semafor.util import serialize_model
import semafor.util
import os
from pathlib import Path
from types import SimpleNamespace
import urllib
from urllib.parse import unquote, urlparse
import cv2
from ultralytics import YOLO
import json
from collections import defaultdict



# We recommend that all loggers have a semafor prefix so that
# we can more easily manage the logging settings.
logger = logging.getLogger("semafor")


def get_detections(results):
    detections = defaultdict(list)
    for result in results:
        if result:
            for box in result:
                label = int(box.boxes.cls[0])
                bbox = box.boxes.xyxy.cpu().numpy()[0]
                classes = get_label(label)
                confidence = box.boxes.conf[0].cpu().numpy()
                score = np.log10(confidence / (1 - confidence + 1e-6))
                d = {'bbox' : bbox,
                        'conf' : confidence,
                        'score': score
                        }
                detections[classes].append(d)
    return detections

def get_label(label):
    if label==0:
        return 'FireOrExplosion'
    elif label == 1:
        return 'PeopleOrGroup'
    elif label in (2,3,4,5,6):
        return 'Person'
    elif label in (7,8,9):
        return 'SignOrWrittenMessaging'
    elif label in (10,11,12,13):
        return 'Symbol'
    elif label in (14,15,16,17,18):
        return 'Vehicles'
    else:
        return 'FirearmOrWeapon'
    
def find_parent_nodes(evigraph: EvidenceGraph, parent_id: str):
    """
    Find all parent nodes for a given child node.
    :param evigraph: the evidence graph.
    :param parent_id: the ID of the child node
    :return: A generator that iterates of the nodes.
    """

    edges = find_incoming_edges(evigraph, parent_id)
    return (evigraphs.find_node(evigraph, e.source) for e in edges)

class DetectObjects(SemaforComponent):
    def __init__(self):
        # use the class name for the logger
        global logger
        logger = logging.getLogger(f"semafor.{self.__class__.__name__}")
        logger.info("Creating the component")
        self.component_spec = None
        SemaforComponent.__init__(self)


    def onInit(self, component):
        # root = logging.getLogger()
        # root.setLevel(logging.DEBUG)

        # hdlr = logger.handlers[0]
        # fmt = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
        # hdlr.setFormatter(fmt)
        # logger.debug("Testing debug logging")

        logger.info("onInit called")
        self.component_spec = component
        self.model = YOLO('manipulated_objects/weights/object_best.pt')
        logger.info("model loaded successfully")

    def onMessage(self, message):
        logger.info("on Message called")
        logger.info(f'Received a message.. ignorning')

    def doRequest(self, message: SemaforMessage) -> SemaforMessage:

        logger.info("do Request called")
        
        if(message.message_type != "ProbeRequestMessage"):
            logger.info(f"It's a {message.message_type} message. I don't care about those so I'm going to ignore it.")
            raise RuntimeError("Component can only process ProbeRequestMessages")
        
        aag = None
        scope = None
        artigraph = message.artigraph
        # logger.info(artigraph)
        evigraph = ev_helper.create_evidence_graph(self, message, artigraph)
        artigraph_id = ev_helper.get_associated_graph_id(evigraph, artigraph, 'ag')

        # logger.info("checking for other action types")
        # logger.info(message.request_action_types.attribution)
        # logger.info(message.request_action_types.characterization)


        if not message.request_action_types.detection:
            # if it is not a detection type request, we are ignoring it
            optout_message = "This probe request is not requesting detection. "\
                "Opting out because other action types are not supported by this analytic."
            logger.info(optout_message)
            opt_out_all(self, artigraph, evigraph, reason="Other", explanation=optout_message)
            return self._sendResponse(evigraph, artigraph, aag, message)

        input_evigraphs = find_analyzed_egs(message, artigraph)
        if len(input_evigraphs):
            input_evigraph = input_evigraphs[0]
            logger.info("Found input evidence graph")
        else: 
            explanation = "Expecting message.evigraphs[0] as input-eg. Opting out."
            logger.info(explanation)
            opt_out_all(self, artigraph, evigraph, reason="Other", explanation=explanation)
            return self._sendResponse(evigraph, artigraph, aag, message)

        # logger.info(input_evigraph)
        
        nodes = artigraph.nodes or []
        edges = artigraph.edges or []
        analytic_scope = cast(AnalyticScope, message.scope)

        logger.info(f"The artigraph's name is {artigraph.metadata.name}. It has {len(nodes)} nodes and {len(edges)} edges.")

        ag_nodes = sorted(artigraph.nodes, key=lambda x: x.node_type)
        ag_nodes_by_type = {
            k: list(g) for k, g in itertools.groupby(ag_nodes, key=lambda x: x.node_type)
        }
        logger.debug(f"This artigraph contains the following node types:")
        for k, v in ag_nodes_by_type.items():
            logger.debug(f" - {k}: {len(v)}")

        if not ag_nodes_by_type["AgImageNode"]:
            opt_out_all(self, artigraph, evigraph, reason="UnsupportedFormat",
                    explanation="Unable to process MMA due to the absence of images.")
            return self._sendResponse(evigraph, artigraph, aag, message)

        input_eg_nodes_by_type = group_nodes_by_type(input_evigraph.nodes)
        evidence_graph.opt_in_modality(evigraph, artigraph, AgImageNode)

        ### checks for input_eg

        manip_image_ref_node = [x for x in
                list(input_eg_nodes_by_type["EvReferenceNode"]) if
                isinstance(x.annotations, dict) and
                isinstance(x.annotations['modality'], str) and
                x.annotations['modality'] == "image"]
        
        if len(manip_image_ref_node) != 1:
            explanation = f"expecting one image reference in EG, but got {len(manip_image_ref_node)}. Opting out."
            logger.info(explanation)
            opt_out_all(self, artigraph, evigraph, reason="UnsupportedFormat", explanation=explanation)
            return self._sendResponse(evigraph, artigraph, aag, message)
        else:
            manip_image_ref_node = manip_image_ref_node[0]
        manip_image_node_id = manip_image_ref_node.referenced_node_id
        manip_image_node = artifact_graph.find_node_in_artifact_graph(artigraph, manip_image_node_id)

        manip_image_fpath = asset.save_file_from_uri_to_sandbox(manip_image_node,
                "/mnt/sandbox/"+message.tid)
        if not os.path.exists(manip_image_fpath):
            raise Exception(f"File not found: {manip_image_fpath}")
        
        logger.info(manip_image_fpath)


        logger.info(f"manip_node_id : {manip_image_node_id}")

        result_dict = dict()
        optout_dict = dict()
        error_dict = dict()

        ext = os.path.splitext(manip_image_fpath)[-1]

        if ext.lower() not in ['.jpeg', '.jpg']:
            # optout_dict[manip_image_node_id] = manip_image_node_id.node_type

            optout_message = "filename not jpg or jpeg."
            logger.info(optout_message)
            opt_out_all(self, artigraph, evigraph, reason="Other", explanation=optout_message)
            return self._sendResponse(evigraph, artigraph, aag, message)
        else:
            try:
                img = Image.open(manip_image_fpath)
                
            except Exception as e:
                logger.error(f'Error to open image {manip_image_node_id} at {manip_image_fpath} '
                                f'The Exception is: {e}')
                error_dict[manip_image_node_id] = e

        if img.get_format_mimetype() != 'image/jpeg':
            optout_message = "Output node : not jpg."
            logger.info(optout_message)
            opt_out_all(self, artigraph, evigraph, reason="Other", explanation=optout_message)
            return self._sendResponse(evigraph, artigraph, aag, message)

        if img.mode != 'RGB':
            optout_message = "outpiut node: not RGB"
            logger.info(optout_message)
            opt_out_all(self, artigraph, evigraph, reason="Other", explanation=optout_message)
            return self._sendResponse(evigraph, artigraph, aag, message)
        
        logger.info("Analytic can read image, printing shape")
        logger.info(img.size)

        logger.info("Sending image for processing")
        ob_results = self.model(img)
        detections = get_detections(ob_results)

        if len(detections) > 1:
            result_dict[manip_image_node_id] = detections

        logger.info(f'The size of result_dict is {len(result_dict)}')
        logger.info(result_dict)
        logger.info(f'Creating evigraph ...')

        if result_dict:
            detection_score_value = max([val[0]['score'] for x in result_dict.values() for key,val in x.items()])
            
            detection_score = EvScore(score_type='LLRScore',
                                      score= 0.94)#detection_score_value  

            # Adding EvDetectionNode 
            detection_node = ev_helper.append_node(evigraph,
                                                   EvDetectionNode,
                                                   score=detection_score)
            ev_helper.append_edge(evigraph,
                                  evigraph.root_node_id,
                                  detection_node.id,
                                  'AnalysisResult')
            
            # Adding EvSemanticConsistencyCheckNode
            consistency_node = ev_helper.append_node(evigraph,
                                                    EvNonSemanticConsistencyCheckNode,
                                                    category=EvSemanticCategory.WHAT_INCONSISTENCY,
                                                    score=detection_score,
                                                    scope=scope,
                                                    )
            
            ev_helper.append_edge(evigraph,
                                detection_node.id,
                                consistency_node.id,
                                'ConsistencyCheck')

            
            manip_polygon = [x.points for x in list(input_eg_nodes_by_type["EvImageLocBoundingPolyNode"])]
            #logger.info(manip_polygon)

            # Adding EvImageLocBoundingNode
            node_bounding_poly = ev_helper.append_node(evigraph,
                    EvImageLocBoundingPolyNode,
                    hole=False,
                    points=manip_polygon)

            
            results = [(key, val[0]['score']) for x in result_dict.values() for key,val in x.items()]

            for entity, score in results:                
                _instance = str(entity)

                # Adding EvConcept node
                concept_node = ev_helper.append_node(evigraph,
                                                    EvConceptNode,
                                                    evidence_type='Entity::'+_instance,
                                                    instance=_instance)
                
                ev_helper.append_edge(evigraph,
                                        consistency_node.id,
                                        concept_node.id,
                                        'Evidence')
                
                
                
                ev_helper.append_edge(evigraph, 
                                      concept_node.id, 
                                      node_bounding_poly.id, 
                                      edge_type="LocationInAsset")


           
            # add the EvReferenceNode
            reference_node = ev_helper.append_node(evigraph,
                                                EvReferenceNode,
                                                referenced_node_id=manip_image_node_id,
                                                referenced_graph_id=artigraph_id)
                
            ev_helper.append_edge(evigraph,
                                node_bounding_poly.id,
                                reference_node.id,
                                'InAsset')

            if message.request_action_types.attribution:
                attrib_score = EvScore(score_type="UndefinedScore", score=0.0)
                attrib_node = append_node(evigraph, EvAttributionNode, score=attrib_score)
                append_edge(evigraph, evigraph.root_node_id, attrib_node.id, "AnalysisResult")
                opt_out(self, artigraph, evigraph, attrib_node, ["node-1"], "NonConformingDomain", explanation="No supported sources")
            
            if message.request_action_types.characterization:
                character_score = EvScore(score_type="UndefinedScore", score=0.0)
                character_node = append_node(evigraph, EvCharacterizationNode, score=character_score)
                append_edge(evigraph, evigraph.root_node_id, character_node.id, "AnalysisResult")
                
                i_node = append_node(evigraph, EvIgnoredAssetNode, reason="Other", explanation="Unsupported analysis type requested")
                append_edge(evigraph, character_node.id, i_node.id, "AnalysisBypass")
    
                append_edge(evigraph, i_node.id, reference_node.id, "IgnoredAsset") 

            for node_id, node_exc in error_dict.items():
                ev_helper.opt_out(self,
                                  artigraph,
                                  evigraph,
                                  detection_node.id,
                                  [f'{node_id}'],
                                  'ErrorProcessingAsset', 
                                  explanation=f'Exception: {node_exc}')

        else:
            logger.info(f"opting out {artigraph.metadata.name}")
            ev_helper.opt_out_all(self,
                                    artigraph,
                                    evigraph,
                                    reason='UnsupportedModality',
                                    explanation='No asset can be processed, opt-out MMA')
                    

        logger.info("Printing evigraph nodes")
        logger.info(evigraph)
            
        with open(f'/mnt/sandbox/{message.tid}/object_detection_eg.json', "w") as f:
            json.dump(serialize_model(evigraph), f, sort_keys=False, indent=4)

        logger.info('Created evigraph.')
        logger.info(f"Successfully processed probe {message.tid}. There are {len(result_dict)} results")


        return self._sendResponse(evigraph, artigraph, aag, message)
    
    def _sendResponse(self, eg: EvidenceGraph, ag: ArtifactGraph, aag, message: SemaforMessage):
        # Writing files to disk for LOCAL (without minikube) debugging
        # evidence_graph.clean_up_evidence_graph(self, eg, ag, reason='Other',
                #explanation='Marking the unused nodes as opt-out before exiting.', logger=logger)

        evidence_graph.save_evidence_graph_to_sandbox(eg, ag, logger=logger,
                sandbox=os.path.join("/mnt/sandbox/", message.tid))

        # The response object should use the same TID as the request message.  The response_action_types field
        # indicates what actions were attempted by this analytic.
        response = ProbeResponseMessage(message_type="ProbeResponseMessage",
            tid=message.tid,
            evigraph=eg,
            augmented_artigraph=aag,
            response_action_types=ActionTypeSpec(detection=True))
        return response


    def onShutdown(self):
        logger.info("onShutdown called")
    
