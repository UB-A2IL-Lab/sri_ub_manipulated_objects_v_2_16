SRI UB Manipulated Object Detector


# Run Project
- docker run -p port:port -v ~/dir:/mnt   -e COMPONENT_SPEC_FILE='component.yaml' image_id
- semafor-test-harness probe  --host http://0.0.0.0:port sri.ub.manipulatedobjects ag/ input_eg/ 

# Customizing this project

When you create a project from this template, you need to modify the `manifest` file to set the properties for your component.  Information can be found at
https://confluence.semaforprogram.com/display/SEM/Tutorial+2B%3A+Creating+a+SemaFor+Component

## Building from source

The published integration-ready component consists of a single artifact: the component docker image, which contains the runtime of the component. This artifact is built using the GNU Make system.  The Makefile defines a number of build targets, which are described in the table below.

| Target | Description |
| ------ | ------ |
| clean | Removes all build artifacts (and any other files not associated with the git repository.  Use this with care if you have local modifications) |
| docker-image | Builds the docker image for the analytic component |
| docker-push | Pushes the docker image to the GitLab registry | 

To build everything, simply run
```
make docker-image
```

## Testing the component

The component can be tested via the test harness.  Refer to https://confluence.semaforprogram.com/display/SEM/Tutorial+2A%3A+Testing+a+Component+With+The+Test+Harness

