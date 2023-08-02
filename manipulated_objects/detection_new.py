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
        self.componentType = "Analytic"
        self.component_spec = None
        SemaforComponent.__init__(self)


    def onInit(self, component):
        logger.info("onInit called")

        self.component_spec = component
        # metadata = component.metadata
        # self.analytic_name = metadata.name

        self.model = YOLO('manipulated_objects/weights/object_best.pt')
        logger.info("model loaded successfully")

    
    def onMessage(self, message):
        logger.info("on Message called")
        logger.info(f'Received a message.. ignorning')


    def doRequest(self, message: SemaforMessage) -> SemaforMessage:
        logger.info("do Request called")
        aag=None
        if (message.message_type != "ProbeRequestMessage"):
            logger.info(f"It's a {message.message_type} message. I don't care about those so I'm going to ignore it.")
            return

        if not message.request_action_types.detection:
            optout_message = "This probe request is not requesting detection. "\
                "Opting out because other action types are not supported by this analytic."
            logger.info(optout_message)
            opt_out_all(self, ag, eg, reason="Other", explanation=optout_message)
            return self._sendResponse(eg, ag, aag, message)

        ag = message.artigraph #: :type artigraph: ArtifactGraph
        eg = create_evidence_graph(self, message, ag)

        #escaping scope chek

        semantic_label = None

        if not message.evigraphs or not message.evigraphs[0]:
            explanation = "Expecting message.evigraphs[0] as input-eg. Opting out."
            logger.info(explanation)
            opt_out_all(self, ag, eg, reason="Other", explanation=explanation)
            return self._sendResponse(eg, ag, aag, message)
        else:
            input_eg = message.evigraphs[0]
            input_eg_id = "input-eg"
            # Copy EvDetection node subtree to add characterization to it
            eg.associated_graphs.append(EgAssociatedGraphReference(local_name="input-eg",
                graph_reference=semafor.util.make_object_reference(input_eg)))
            input_eg_det_node = evidence_graph.find_score_nodes(input_eg)["Detection"]
            if not input_eg_det_node:
                explanation = "No Detection node found in input EG. Opting out."
                logger.info(explanation)
                opt_out_all(self, ag, eg, reason="Other", explanation=explanation)
                return self._sendResponse(eg, ag, aag, message)
        
        
        logger.info(f"Identified semantic label: {semantic_label}")

        ag_id = get_associated_graph_id(eg, ag)
        nodes = ag.nodes or []
        nodes_by_type = artifact_graph.group_nodes_by_modality(nodes)
        logger.info(f"Found {sum([len(v) for v in nodes_by_type.values()])} media assets.")

            
        # opt-out all if no image assets are present, continue otherwise
        if not nodes_by_type["AgImageNode"]:
            opt_out_all(self, ag, eg, reason="UnsupportedFormat",
                    explanation="Unable to process MMA due to the absence of images.")
            return self._sendResponse(eg, ag, aag, message)

        # opt out of everything but images
        evidence_graph.opt_in_modality(eg, ag, AgImageNode)


        input_eg_nodes_by_type = group_nodes_by_type(input_eg.nodes)
        if not input_eg_nodes_by_type.get("EvDetectionNode"):
            explanation = "Missing one of EvDetectionNode and ImageLocBoundingPolyNode in input EG. Opting out."
            logger.info(explanation)
            opt_out_all(self, ag, eg, reason="UnsupportedFormat", explanation=explanation)
            return self._sendResponse(eg, ag, aag, message)
        if semantic_label and not input_eg_nodes_by_type.get("EvImageLocBoundingPolyNode"):
            explanation = "Missing ImageLocBoundingPolyNode in input EG, when semantic_label was provided in scope. Opting out."
            logger.info(explanation)
            opt_out_all(self, ag, eg, reason="UnsupportedFormat", explanation=explanation)

        if input_eg_nodes_by_type.get("EvImageLocBoundingPolyNode"):
            manip_polygon_node = list(input_eg_nodes_by_type.get("EvImageLocBoundingPolyNode"))[0]
            # Assuming there's just one incoming edge for each node.
            input_concept_node = list(find_parent_nodes(input_eg, manip_polygon_node.id))[0]
            input_eg_cons_node = list(find_parent_nodes(input_eg, input_concept_node.id))[0]
        else: 
            manip_polygon_node = None
            input_concept_node = None
            input_eg_cons_node = None

        if semantic_label and not manip_polygon_node:
            explanation = "scope had a semanticLabels request, but no EvImageLocBoundingPolyNode found."
            opt_out_all(self, ag, eg, reason="UnsupportedFormat", explanation=explanation)
            return self._sendResponse(eg, ag, aag, message)
        # Actual Eval 3.2.3.1-6 case. One semantic label of interest, manip polygon given.
        elif semantic_label and manip_polygon_node:
            case = "3.2.3.1-6"
            if input_eg_nodes_by_type.get("EvConceptNode"):
                input_concept_node = [x for x in
                    list(input_eg_nodes_by_type.get("EvConceptNode"))][0]
            else:
                explanation = f"expecting one image manipulation concept node in EG, but got zero. Opting out."
                logger.info(explanation)
                opt_out_all(self, ag, eg, reason="UnsupportedFormat", explanation=explanation)
                return self._sendResponse(eg, ag, aag, message)
            if len(input_concept_node) != 1:
                explanation = f"expecting one image manipulation concept node in EG, but got {len(input_concept_node)}. Opting out."
                logger.info(explanation)
                opt_out_all(self, ag, eg, reason="UnsupportedFormat", explanation=explanation)
                return self._sendResponse(eg, ag, aag, message)
            else:
                input_concept_node = input_concept_node[0]
            input_eg_cons_node = list(find_parent_nodes(input_eg, input_concept_node.id))[0]
        #Eval 4.1.1c case. Manip poly given. Provide ConceptNodes for all spliced objects found.
        elif manip_polygon_node and not semantic_label:
            case = "4.1.1c"
            if input_eg_nodes_by_type.get("EvConceptNode"):
                input_concept_node = [x for x in
                    list(input_eg_nodes_by_type.get("EvConceptNode"))][0]
            else:
                explanation = f"expecting one image manipulation concept node in EG, but got zero. Opting out."
                logger.info(explanation)
                opt_out_all(self, ag, eg, reason="UnsupportedFormat", explanation=explanation)
                return self._sendResponse(eg, ag, aag, message)
            input_eg_cons_node = list(find_parent_nodes(input_eg, input_concept_node.id))[0]
        # Eval 3.2.3.7 case: input EG has no AomLocIdNode. localization only
        else:
            case = "3.2.3.7"
            if input_eg_nodes_by_type.get("EvConceptNode"):
                input_concept_node = [x for x in
                    list(input_eg_nodes_by_type.get("EvConceptNode")) if
                    isinstance(x.instance, str) and
                    x.instance == "image Manipulation"]
            else:
                explanation = f"expecting one image manipulation concept node in EG, but got zero. Opting out."
                logger.info(explanation)
                opt_out_all(self, ag, eg, reason="UnsupportedFormat", explanation=explanation)
                return self._sendResponse(eg, ag, aag, message)
            if len(input_concept_node) != 1:
                explanation = f"expecting one image manipulation concept node in EG, but got {len(input_concept_node)}. Opting out."
                logger.info(explanation)
                opt_out_all(self, ag, eg, reason="UnsupportedFormat", explanation=explanation)
                return self._sendResponse(eg, ag, aag, message)
            else:
                input_concept_node = input_concept_node[0]
            input_eg_cons_node = list(find_parent_nodes(input_eg, input_concept_node.id))[0]

        manip_image_ref_node = [x for x in
            list(input_eg_nodes_by_type["EvReferenceNode"]) if
            isinstance(x.annotations, dict) and
            isinstance(x.annotations['modality'], str) and
            x.annotations['modality'] == "image"]
        if len(manip_image_ref_node) != 1:
            explanation = f"expecting one image reference in EG, but got {len(manip_image_ref_node)}. Opting out."
            logger.info(explanation)
            opt_out_all(self, ag, eg, reason="UnsupportedFormat", explanation=explanation)
            return self._sendResponse(eg, ag, aag, message)
        else:
            manip_image_ref_node = manip_image_ref_node[0]
        manip_image_node_id = manip_image_ref_node.referenced_node_id
        manip_image_node = artifact_graph.find_node_in_artifact_graph(ag, manip_image_node_id)
 
        manip_image_fpath = asset.save_file_from_uri_to_sandbox(manip_image_node,
                "/mnt/sandbox/"+message.tid)
        if not os.path.exists(manip_image_fpath):
            raise Exception(f"File not found: {manip_image_fpath}")
        
        logger.info(manip_image_fpath)
        logger.info(f"manip_node_id : {manip_image_node_id}")

        ### got image path , performing detections
        result_dict = dict()
        optout_dict = dict()
        error_dict = dict()

        ext = os.path.splitext(manip_image_fpath)[-1]

        if ext.lower() not in ['.jpeg', '.jpg']:
            # optout_dict[manip_image_node_id] = manip_image_node_id.node_type
            optout_message = "filename not jpg or jpeg."
            logger.info(optout_message)
            opt_out_all(self, ag, eg, reason="Other", explanation=optout_message)
            return self._sendResponse(eg, ag, aag, message)
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
            opt_out_all(self, ag, eg, reason="Other", explanation=optout_message)
            return self._sendResponse(eg, ag, aag, message)

        if img.mode != 'RGB':
            optout_message = "outpiut node: not RGB"
            logger.info(optout_message)
            opt_out_all(self, ag, eg, reason="Other", explanation=optout_message)
            return self._sendResponse(eg, ag, aag, message)
        
        logger.info("Analytic can read image, printing shape")
        logger.info(img.size)

        logger.info("Sending image for processing")
        ob_results = self.model(img)
        detections = get_detections(ob_results)

        #### detection completed

        if len(detections) > 1:
            result_dict[manip_image_node_id] = detections

        results = [(key, val[0]['score']) for x in result_dict.values() for key,val in x.items()]
        detected_object_types = []
        for entity, score in results:
            detected_object_types.append(str(entity))

        manip_polygon_pred = None
        llr_score = EvScore(score=1.0, score_type="LLRScore")

        logger.info(f'The count of detections is {len(detected_object_types)}')
        logger.info(detected_object_types)
        logger.info(f'Creating evigraph ...')


        logger.info(f"Task  : {case}")
        logger.info(f"Input eg id : {input_eg_id}")
        logger.info(f"ag id : {ag_id}")

        node_detection = append_node(eg,
                EvDetectionNode,
                score=EvScore(score=0, score_type="UndefinedScore")
                )
        append_edge(eg, eg.root_node_id, node_detection.id, edge_type="AnalysisResult")

        if case == "4.1.1c":
            #Object detection consistency node
            node_consistency = append_node(eg,
                    EvSemanticConsistencyCheckNode,
                    score=llr_score,
                    scope=cast(AnalyticScope, message.scope),
                    category=EvSemanticCategory.WHAT_INCONSISTENCY)
            append_edge(eg, node_detection.id, node_consistency.id, edge_type="ConsistencyCheck")
        else:
            
            node_consistency = append_node(eg,
                    EvNonSemanticConsistencyCheckNode,
                    score=llr_score,
                    category=EvSemanticCategory.DATADRIVEN_INCONSISTENCY)
            append_edge(eg, node_detection.id, node_consistency.id, edge_type="ConsistencyCheck")

        ev_input_eg_reference = append_node(eg,
                EvReferenceNode,
                referenced_graph_id=input_eg_id, referenced_node_id=input_eg_cons_node.id
                )
        ev_input_eg_reference.annotations = {
                'modality': None,
                'asset': None
                }
        append_edge(eg, node_consistency.id, ev_input_eg_reference.id, edge_type="Evidence")

        node_model = append_node(eg,
                EvAnalyticModelNode,
                model_type = "manipulationModel",
                model_name = "Object detector/manipulation model")
        append_edge(eg, node_consistency.id, node_model.id, edge_type="Evidence")


        if case == "4.1.1c":
            object_type_mapping = {
                    "Person": "Entity::Person",
                    "People": "Entity::PeopleOrGroup",
                    "Fire": "Entity::FireOrExplosion",
                    "Symbol": "Entity::Symbol",
                    "Firearms": "Entity::FirearmOrWeapon",
                    "Signs": "Entity::SignOrWrittenMessaging",
                    "Vehicles": "Entity::Vehicles"
                    }
            
            if detected_object_types:
                node_bounding_poly = append_node(eg,
                        EvImageLocBoundingPolyNode,
                        hole = False,
                        points = manip_polygon_node.points)

                node_reference = append_node(eg, EvReferenceNode,
                        referenced_graph_id=ag_id, referenced_node_id=manip_image_node_id)
                
                append_edge(eg, node_bounding_poly.id, node_reference.id, edge_type="InAsset")
        
        for obj_type in detected_object_types:
            node_obj_concept = append_node(eg,
                    EvConceptNode,
                    evidence_type=object_type_mapping[obj_type])
            
            append_edge(eg, node_consistency.id, node_obj_concept.id, edge_type="Evidence")

            append_edge(eg, node_obj_concept.id, node_bounding_poly.id, edge_type="LocationInAsset")

        logger.info(f'manip_polygon_pred: {manip_polygon_pred}')


        logger.info('Created evigraph.')
        logger.info(f"Successfully processed probe {message.tid}. There are {len(result_dict)} results")
        logger.info('Printing evigraph.')
        logger.info(eg)
        return self._sendResponse(eg, ag, aag, message)

    def _sendResponse(self, eg: EvidenceGraph, ag: ArtifactGraph, aag, message: SemaforMessage):

        evidence_graph.save_evidence_graph_to_sandbox(eg, ag, logger=logger,
                sandbox=os.path.join("/mnt/sandbox/", message.tid))

        response = ProbeResponseMessage(message_type="ProbeResponseMessage",
            tid=message.tid,
            evigraph=eg,
            augmented_artigraph=aag,
            response_action_types=ActionTypeSpec(detection=True))
        return response

    def onShutdown(self):
        logger.info("onShutdown called")
