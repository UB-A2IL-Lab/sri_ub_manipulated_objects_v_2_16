kind: Component
metadata:
  name: sri-ub-manipulatedobjects
  summary: This analytic detects objects within manipualted regions
  description: |
    This analytic detects objects within manipualted regions
image: registry.semaforprogram.com/semafor/teams/ta1/sri/buffalo/sri.ub.manipulatedobjects:0.2.16
version: 0.2.16
requestTypes: [Probe]
config:
  logLevel: info
  properties:
    componentType: analytic
    throwErrorOnMessage: False
    # evalMode: raw
    # taskMode: manip_synth
computeResources:
  gpu: 1
poc:
  name: Abhishek Kumar
  email: akumar58@buffalo.edu
repositoryUrl: https://gitlab.semaforprogram.com/semafor/teams/ta1/sri/buffalo/sri.ub.manipulatedobjects.git
# externalResources:
#   dvc:
#     path: resources/0.3.0
internetUsage:
  usesInternet: False
  description: N/A
maturity: Experimental
supportedAssetTypes:
  - Image
supportedActionType:
  detection: True
  attribution: False
  characterization: False
  fusion: False
  prioritization: False
scope:
consistencyCheckTypes:
  - NonSemantic

analyticCapabilities:
  - taskName: image_manipulation_label_detection
    scope:
      - parameter: supportedEvidence
        choices:
          - Entity::Person
          - Entity::PeopleOrGroup
          - Entity::Vehicles
          - Entity::FireOrExplosion
          - Entity::Symbol
          - Entity::FirearmOrWeapon
          - Entity::SignOrWrittenMessaging
      - parameter: fileFormat
        choices: [ jpg, png ]
