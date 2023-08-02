#  .▄▄ · ▄▄▄ .• ▌ ▄ ·.  ▄▄▄· ·▄▄▄      ▄▄▄    #
#  ▐█ ▀. ▀▄.▀··██ ▐███▪▐█ ▀█ ▐▄▄·▪     ▀▄ █·  #
#  ▄▀▀▀█▄▐▀▀▪▄▐█ ▌▐▌▐█·▄█▀▀█ ██▪  ▄█▀▄ ▐▀▀▄   #
#  ▐█▄▪▐█▐█▄▄▌██ ██▌▐█▌▐█ ▪▐▌██▌.▐█▌.▐▌▐█•█▌  #
#   ▀▀▀▀  ▀▀▀ ▀▀  █▪▀▀▀ ▀  ▀ ▀▀▀  ▀█▄▀▪.▀  ▀  #

# This is the dockerfile for creating the docker image of the Semafor Component 

# The first line of your analytic's Dockerfile contains the starting base image.
# In this case, we select a base image of python version 3.8 (see
# https://hub.docker.com/_/python)
# ARG REGISTRY_OVERRIDE=docker.io
# FROM ${REGISTRY_OVERRIDE}/python:3.8
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Some configuration settings to get pip working with our gitlab instance
ARG https_proxy
ARG GL_TOK=sER6QxVVbfZLXnk3fQYv
ARG GL_API=https://__token__:$GL_TOK@gitlab.semaforprogram.com/api/v4/projects
RUN pip config set global.extra-index-url "${GL_API}/2/packages/pypi/simple \
                                           ${GL_API}/3/packages/pypi/simple"
RUN pip config set global.trusted-host "pypi.org pypi.python.org \
                                        files.pythonhosted.org \
                                        gitlab.semaforprogram.com"

# Install dependecies for opencv
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libgl1 python3-opencv libgl1-mesa-glx -y
# python3-opencv
RUN apt update && apt install -y git
# RUN apt-get install git -y 

# Install dependencies from requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

RUN pip install simplejson
RUN pip install -U iopath
RUN pip install psutil
RUN pip install opencv-python
# RUN pip install pytorchvideo
# RUN pip install -U 'git+https://github.com/facebookresearch/fvcore.git'
# RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# RUN pip install "git+https://github.com/facebookresearch/pytorchvideo.git"
# RUN pip install fairscale

# Install the semafor toolkit and service library
ARG TOOLKIT_VERSION
ARG SERVICE_VERSION
ARG AAG_GENERATOR_VERSION
RUN pip install semafor-toolkit${TOOLKIT_VERSION:+==$TOOLKIT_VERSION}
RUN pip install semafor-service${SERVICE_VERSION:+==$SERVICE_VERSION}
# RUN pip install aag-generator${AAG_GENERATOR_VERSION:+==$AAG_GENERATOR_VERSION}

# ARG AOMSDK_VERSION
# ARG SIDUTILS_VERSION
# RUN pip install semafor-aom-sdk${AOMSDK_VERSION:+==$AOMSDK_VERSION}
# RUN pip install sid-utils${SIDUTILS_VERSION:+==$SIDUTILS_VERSION}

# SID UTILS
RUN pip install sid-utils -U --trusted-host gitlab.semaforprogram.com --extra-index-url \
   https://__token__:XisVP5d-X7j_KziFuZy7@gitlab.semaforprogram.com/api/v4/projects/10/packages/pypi/simple


# Now, install our source code.  We do this close to the end to maximize the
# amount of layer caching Docker can make use of assuming that you are typically
# making changes to the analytic runtime source code.  You can put the code
# wherever you want, but it needs to be on the python path or in the working
# directory.
COPY . /app
WORKDIR /app

# Configure the entry point to the container and the port that the service will
# listen on.
ARG COMPONENT_CLASS
ENV COMPONENT_CLASS=$COMPONENT_CLASS
CMD python -m semafor_service $COMPONENT_CLASS 8080
EXPOSE 8080