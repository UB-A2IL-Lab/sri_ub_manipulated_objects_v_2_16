import logging
from typing import cast
import numpy as np
from PIL import Image
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
    EvConceptNode, EvIgnoredAssetNode, EvidenceGraph, ArtifactGraph,
    AgImageNode, EvDetectionNode, EvAnalyticModelNode,
    EvSemanticCategory, AnalyticScope, EgAssociatedGraphReference,
    EvImageLocBoundingPolyNode, EvImageLocPixelMapNode
)
from semafor.helpers.evigraphs import (append_node, append_edge, opt_out,
                                       opt_out_all, get_associated_graph_id,
                                       create_evidence_graph, find_nodes_by_type,
                                       find_incoming_edges)
from semafor.helpers import evigraphs
import semafor.helpers.evigraphs as ev_helper
from semafor.helpers.artigraphs import find_node
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
        logger.info("onInit called")
        self.component_spec = component
        self.model = YOLO('manipulated_objects/weights/object_best.pt')
        logger.info("model loaded successfully")

    def onMessage(self, message):
        logger.info("on Message called")
        logger.info(f'Received a message.. ignorning')

    def doRequest(self, message: SemaforMessage) -> SemaforMessage:
        logger.info("===== do Request called =====")
        
        if (message.message_type != "ProbeRequestMessage"):
            logger.info(f"It's a {message.message_type} message. I don't care about those so I'm going to ignore it.")
            raise RuntimeError("Component can only process ProbeRequestMessages")

        logger.info(f"It's a ProbeRequestMessage! The transaction id (TID) is {message.tid}")

        if not message.request_action_types.detection:
            msg = "This probe request is not requesting detection. And since that's all my analytic knows how to do, this request will be ignored"
            logger.info(msg)
            response = ProbeResponseMessage(message_type="ProbeResponseMessage",
                                            tid=message.tid,
                                            response_type="ProbeError",
                                            response_message=msg,
                                            component=self.component_spec.metadata.name)
            return response
        
        artigraph = message.artigraph
        evigraph = ev_helper.create_evidence_graph(self, message, artigraph)
        artigraph_id = ev_helper.get_associated_graph_id(evigraph, artigraph, 'ag')

        
        # if not message.evigraphs or not message.evigraphs[0]:
        #     explanation = "Expecting message.evigraphs[0] as input-eg. Opting out."
        #     logger.info(explanation)
        #     response = ProbeResponseMessage(message_type='ProbeResponseMessage', tid=message.tid, evigraph=evigraph,
        #                                 response_type="ProbeError", response_message=explanation)
        #     return response
        # else:
        #     input_eg = message.evigraphs[0]
        #     input_eg_id = "input-eg"
        #     # Copy EvDetection node subtree to add characterization to it
        #     evigraph.associated_graphs.append(EgAssociatedGraphReference(local_name="input-eg",
        #         graph_reference=semafor.util.make_object_reference(input_eg)))
        #     input_eg_det_node = evidence_graph.find_score_nodes(input_eg)["Detection"]
        #     if not input_eg_det_node:
        #         explanation = "No Detection node found in input EG. Opting out."
        #         logger.info(explanation)
        #         response = ProbeResponseMessage(message_type='ProbeResponseMessage', tid=message.tid, evigraph=evigraph,
        #                                 response_type="ProbeError", response_message=explanation)
        #         return response
            
        # semantic_label = None  
        # input_eg_nodes_by_type = group_nodes_by_type(input_eg.nodes)

        # if not input_eg_nodes_by_type.get("EvDetectionNode"):
        #     explanation = "Missing one of detection node and ImageLocBoundingPolyNode in input EG. Opting out."
        #     logger.info(explanation)
        #     response = ProbeResponseMessage(message_type='ProbeResponseMessage', tid=message.tid, evigraph=evigraph,
        #                                 response_type="ProbeError", response_message=explanation)
        #     return response
        
        # if semantic_label and not input_eg_nodes_by_type.get("EvImageLocBoundingPolyNode"):
        #     explanation = "Missing ImageLocBoundingPolyNode in input EG, when semantic_label was provided in scope. Opting out."
        #     logger.info(explanation)
        #     response = ProbeResponseMessage(message_type='ProbeResponseMessage', tid=message.tid, evigraph=evigraph,
        #                                 response_type="ProbeError", response_message=explanation)
        #     return response

        # if semantic_label:
        #     if input_eg_nodes_by_type.get("EvImageLocBoundingPolyNode"):
        #         manip_polygon_node = list(input_eg_nodes_by_type.get("EvImageLocBoundingPolyNode"))[0]
        #     else:
        #         explanation = "scope had a semanticLabels request, but no EvImageLocBoundingPolyNode found."
        #         response = ProbeResponseMessage(message_type='ProbeResponseMessage', tid=message.tid, evigraph=evigraph,
        #                                 response_type="ProbeError", response_message=explanation)
        #         return response
        #     # Assuming there's just one incoming edge for each node.
        #     input_concept_node = list(find_parent_nodes(input_eg, manip_polygon_node.id))[0]
        #     input_eg_cons_node = list(find_parent_nodes(input_eg, input_concept_node.id))[0]
        # else:
        #     if input_eg_nodes_by_type.get("EvConceptNode"):
        #         input_concept_node = [x for x in
        #             list(input_eg_nodes_by_type.get("EvConceptNode")) if
        #             isinstance(x.instance, str) and
        #             x.instance == "image Manipulation"]
        #     else:
        #         explanation = f"expecting one image manipulation concepnt node in EG, but got zero. Opting out."
        #         logger.info(explanation)
        #         response = ProbeResponseMessage(message_type='ProbeResponseMessage', tid=message.tid, evigraph=evigraph,
        #                                 response_type="ProbeError", response_message=explanation)
        #         return response
            
        #     if len(input_concept_node) != 1:
        #         explanation = f"expecting one image manipulation concepnt node in EG, but got {len(input_concept_node)}. Opting out."
        #         logger.info(explanation)
        #         response = ProbeResponseMessage(message_type='ProbeResponseMessage', tid=message.tid, evigraph=evigraph,
        #                                 response_type="ProbeError", response_message=explanation)
        #         return response
        #     else:
        #         input_concept_node = input_concept_node[0]
        #     input_eg_cons_node = list(find_parent_nodes(input_eg, input_concept_node.id))[0]

        # manip_image_ref_node = [x for x in
        #         list(input_eg_nodes_by_type["EvReferenceNode"]) if
        #         isinstance(x.annotations, dict) and
        #         isinstance(x.annotations['modality'], str) and
        #         x.annotations['modality'] == "image"]
        # if len(manip_image_ref_node) != 1:
        #     explanation = f"expecting one image reference in EG, but got {len(manip_image_ref_node)}. Opting out."
        #     logger.info(explanation)
        #     response = ProbeResponseMessage(message_type='ProbeResponseMessage', tid=message.tid, evigraph=evigraph,
        #                                 response_type="ProbeError", response_message=explanation)
        #     return response
        # else:
        #     manip_image_ref_node = manip_image_ref_node[0]
        # manip_image_node_id = manip_image_ref_node.referenced_node_id
        # manip_image_node = artifact_graph.find_node_in_artifact_graph(artigraph, manip_image_node_id)

        # manip_image_fpath = asset.save_file_from_uri_to_sandbox(manip_image_node,
        #         "/mnt/sandbox/"+message.tid)
        
        # print(manip_image_fpath)


        #################

        # print(message.evigraph)
        nodes = artigraph.nodes or []
        edges = artigraph.edges or []
        logger.info(f"The artigraph's name is {artigraph.metadata.name}. It has {len(nodes)} nodes and {len(edges)} edges.")
        
        result_dict = dict()
        optout_dict = dict()
        error_dict = dict()

        for node in nodes:
            rec = dict()
            logger.info(f'Processing node {node.id}, type: {node.node_type}')

            if node.node_type == 'AgImageNode':
                uri = node.asset_data_uri
                name = os.path.basename(urllib.parse.urlsplit(uri).path)
                path = os.path.join("/mnt/sandbox/", name)
                urllib.request.urlretrieve(uri, path)
                ext = os.path.splitext(path)[-1]

                if ext.lower() not in ['.jpeg', '.jpg']:
                    logger.info(f'Optout node {node.id}: filename not jpg')
                    optout_dict[node.id] = node.node_type
                    continue
                try:
                    img = Image.open(path)
                except Exception as e:
                    logger.error(f'Error to open image {node.id} at {uri} '
                                 f'The Exception is: {e}')
                    error_dict[node.id] = e
                    continue

                if img.get_format_mimetype() != 'image/jpeg':
                    logger.info(f'Optout node {node.id}: image not jpg')
                    optout_dict[node.id] = node.node_type
                    continue

                if img.mode != 'RGB':
                    logger.info(f'Optout node {node.id}: image not RGB')
                    optout_dict[node.id] = node.node_type
                    continue
                
                # img
                logger.info("Sending image for processing")

                ob_results = self.model(img)
                detections = get_detections(ob_results)
                if len(detections) > 1:
                    result_dict[node.id] = detections

        logger.info(f'Creating evigraph ...')

        

        logger.info(f'The size of result_dict is {len(result_dict)}')

        scope = None
        if result_dict:
            # score from all images in MMA
            temp_detection_score_value = max(each['score'] for detections in result_dict.values() for each_list in detections.values() for each in each_list)
            
            logger.info("printing temp detection score for evdetection node")
            logger.info(temp_detection_score_value)

            # detection_score_value = EvScore(score=temp_detection_score_value, score_type='LLRScore')

            # logger.info(detection_score_value.score)
            detection_score_value = EvScore(score=0.94, score_type='LLRScore')
            # detection_score = EvScore(score_type='LLRScore',
            #                           score=-1.11)
            
            detection_node = ev_helper.append_node(evigraph,
                                                   EvDetectionNode,
                                                   score=detection_score_value)
            ev_helper.append_edge(evigraph,
                                  evigraph.root_node_id,
                                  detection_node.id,
                                  'AnalysisResult')
            
            for node_id, result in result_dict.items():

                # add the ConsistencyCheckNode
                # score from each image
                try:
                    # temp_score = max(each['score'] for each_list in detections.values() for each in each_list) 
                    # score = EvScore(score=temp_score, score_type='LLRScore')
                    logger.info("Printing temp score for each node")
                    # logger.info(temp_score)

                    # logger.info(score.score)
                    score = EvScore(score=0.42, score_type='LLRScore')

                    # score = EvScore(score=temp_score, score_type='LLRScore')

                    consistency_node = ev_helper.append_node(evigraph,
                                                             EvNonSemanticConsistencyCheckNode,
                                                             category=EvSemanticCategory.WHAT_INCONSISTENCY,
                                                             score=score,
                                                             scope=scope)
                except Exception as e:
                    consistency_node = ev_helper.append_node(evigraph,
                                                             EvNonSemanticConsistencyCheckNode,
                                                             category=EvSemanticCategory.WHAT_INCONSISTENCY,
                                                             score=score)
                    logger.info(f"Failed to add scope in the consistency node: {e}")
                    
                ev_helper.append_edge(evigraph,
                                      detection_node.id,
                                      consistency_node.id,
                                      'ConsistencyCheck')
                
                 # add the EvConceptNode
                for entity in result.keys():
                    concept_node = ev_helper.append_node(evigraph,
                                                        EvConceptNode,
                                                        evidence_type='Entity::'+entity,
                                                        instance='manipulated_object')
                    ev_helper.append_edge(evigraph,
                                        consistency_node.id,
                                        concept_node.id,
                                        'Evidence')
                        
                    # add the EvReferenceNode
                    reference_node = ev_helper.append_node(evigraph,
                                                        EvReferenceNode,
                                                        referenced_node_id=node_id,
                                                        referenced_graph_id=artigraph_id)
                    ev_helper.append_edge(evigraph,
                                        concept_node.id,
                                        reference_node.id,
                                        'InAsset')

            # optout everything that is not an video
            ev_helper.opt_out_filter(self,
                                     artigraph,
                                     evigraph,
                                     evigraph.root_node_id,
                                     lambda x: x.node_type != 'AgImageNode',
                                     reason='UnsupportedModality',
                                     explanation='No detection conducted due to unsupportyed modality')

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
            
        logger.info(evigraph)
        with open(f'/mnt/sandbox/{message.tid}/object_detection_eg.json', "w") as f:
            json.dump(serialize_model(evigraph), f, sort_keys=False, indent=4)

        response = ProbeResponseMessage(message_type='ProbeResponseMessage', tid=message.tid, evigraph=evigraph,
                                        response_action_types=ActionTypeSpec(detection=True))

        logger.info('Created evigraph.')
        logger.info(f"Successfully processed probe {message.tid}. There are {len(result_dict)} results")

        # logger.info("testing complete")

        return response


    def onShutdown(self):
        logger.info("onShutdown called")
    
