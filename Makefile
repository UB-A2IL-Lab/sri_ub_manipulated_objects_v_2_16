# # Load the manifest as environment variables

include manifest
export

REGISTRY_OVERRIDE:=registry.semaforprogram.com/semafor/lib/external
GOMPLATE:=$(REGISTRY_OVERRIDE)/hairyhenderson/gomplate:v3.8.0-slim

.PHONY:

ifeq ($(OS),Windows_NT)
clean:
	if exist chart\Chart.yaml del chart\Chart.yaml
	if exist chart\values.yaml del chart\values.yaml
	if exist component.yaml del component.yaml
else
clean:
	$(unlink) chart/Chart.yaml chart/values.yaml component.yaml
endif

ifeq ($(OS),Windows_NT)
component.yaml: component.yaml.tpl manifest
	cmd /c docker run --rm -v %CD%:/data \
		--env-file manifest $(GOMPLATE) \
		-f /data/component.yaml.tpl \
		-o /data/component.yaml
else
component.yaml: component.yaml.tpl manifest
	docker run --rm -v $(PWD):/data --env-file manifest \
	  $(GOMPLATE) -f /data/component.yaml.tpl -o /data/component.yaml
endif

docker-image: component.yaml
	docker build . -t $(COMPONENT_DOCKER_REPO):$(COMPONENT_VERSION) \
	--build-arg TOOLKIT_VERSION --build-arg SERVICE_VERSION \
	--build-arg COMPONENT_CLASS --build-arg https_proxy \
	--build-arg REGISTRY_OVERRIDE

docker-push: docker-image
	docker push $(COMPONENT_DOCKER_REPO):$(COMPONENT_VERSION)

ifeq ($(OS),Windows_NT)
chart\Chart.yaml: chart/Chart.yaml.tpl Makefile manifest
	cmd /c docker run --rm -v %CD%\chart:/data --env-file manifest \
	  $(GOMPLATE) -f /data/Chart.yaml.tpl -o /data/Chart.yaml
chart\values.yaml: chart/values.yaml.tpl Makefile manifest
	cmd /c docker run --rm -v %CD%\chart:/data --env-file manifest \
	  $(GOMPLATE) -f /data/values.yaml.tpl -o /data/values.yaml
else
chart/Chart.yaml: chart/Chart.yaml.tpl Makefile manifest
	docker run --rm -v $(PWD)/chart:/data --env-file manifest \
	  $(GOMPLATE) -f /data/Chart.yaml.tpl -o /data/Chart.yaml
chart/values.yaml: chart/values.yaml.tpl Makefile manifest
	docker run --rm -v $(PWD)/chart:/data --env-file manifest \
	  $(GOMPLATE) -f /data/values.yaml.tpl -o /data/values.yaml
endif

ifeq ($(OS),Windows_NT)
helm-image: chart\Chart.yaml chart\values.yaml
	cmd /c "(echo FROM scratch & echo COPY . $(COMPONENT_NAME_DASHIFIED)) | docker build -t $(COMPONENT_CHART_REPO):${COMPONENT_CHART_VERSION} chart -f -"
else
helm-image: chart/Chart.yaml chart/values.yaml
	printf "FROM scratch\nCOPY . $(COMPONENT_NAME_DASHIFIED)" | \
	docker build -t $(COMPONENT_CHART_REPO):${COMPONENT_CHART_VERSION} chart -f -
endif

helm-push: helm-image
	docker push $(COMPONENT_CHART_REPO):${COMPONENT_CHART_VERSION}