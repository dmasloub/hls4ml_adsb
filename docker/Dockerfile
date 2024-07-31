FROM jupyter/tensorflow-notebook:tensorflow-2.11.1

# Install prequisites
USER root
RUN apt-get update -y && \
    apt-get install --no-install-recommends -y \
    curl \
    libtinfo5 \
    libc6-dev-i386 \
    net-tools \
    graphviz \
    make \
    unzip \
    g++ \
    xvfb \
    git \
    libncursesw5 \
    libc6-dev-i386 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install additional system dependencies (if any)
# RUN apt-get install --no-install-recommends -y <additional-package>

# Switch to the default user
USER ${NB_USER}

# Install Python dependencies using mamba (conda)
RUN mamba install -y -c conda-forge \
    graphviz==7.1.0 \
    pydot==1.4.2 \
    tensorflow-datasets==4.8.3 \
    jupyter-book==0.15.1 \
    jupyter_contrib_nbextensions==0.7.0

# Install additional Python libraries using pip
RUN pip install \
    hls4ml[profiling]==0.8.0 \
    qkeras==0.9.0 \
    conifer==0.2b0 \
    pysr==0.16.3 \
    tsfresh==0.20.2 \
    stumpy==1.12.0 
    # Add your project-specific pip packages here

# Clean up
RUN mamba clean --all -f -y && \
    mamba list && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Copy your project files into the container
COPY . /home/${NB_USER}/adsb_pynqz2

# Set working directory
WORKDIR /home/${NB_USER}/adsb_pynqz2

# Fix permissions for the working directory
USER root
RUN chown -R 1000:100 /home/${NB_USER}/adsb_pynqz2

USER ${NB_USER}

# Run any setup scripts if needed
# RUN ./setup.sh

LABEL org.opencontainers.image.source https://github.com/dmasloub/hls4ml_adsb