ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS base
RUN if [ -f /etc/apt/apt.conf.d/proxy.conf ]; then rm /etc/apt/apt.conf.d/proxy.conf; fi && \
    if [ ! -z ${HTTP_PROXY} ]; then echo "Acquire::http::Proxy \"${HTTP_PROXY}\";" >> /etc/apt/apt.conf.d/proxy.conf; fi && \
    if [ ! -z ${HTTPS_PROXY} ]; then echo "Acquire::https::Proxy \"${HTTPS_PROXY}\";" >> /etc/apt/apt.conf.d/proxy.conf; fi
RUN apt update && \
    apt full-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y \
    sudo \
    ca-certificates \
    git \
    curl \
    wget \
    vim \
    numactl \
    gcc-12 \
    g++-12 \
    make
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100
RUN apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    if [ -f /etc/apt/apt.conf.d/proxy.conf ]; then rm /etc/apt/apt.conf.d/proxy.conf; fi

RUN useradd -m ubuntu && \
    echo 'ubuntu ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu

RUN curl -fsSL -v -o miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    bash miniconda.sh -b -p ./miniconda3 && \
    rm miniconda.sh && \
    echo "source ~/miniconda3/bin/activate" >> ./.bashrc

FROM base AS dev
ARG COMPILE
COPY --chown=ubuntu:ubuntu . ./kana
RUN . ./miniconda3/bin/activate && \
    conda create -y -n compile_py310 python=3.10 && conda activate compile_py310 && \
    pip config set global.index-url https://mirrors.sustech.edu.cn/pypi/web/simple && \
    cd kana/thirdparty/intel-extension-for-pytorch/examples/cpu/inference/python/llm && \
    if [ -z ${COMPILE} ]; then bash tools/env_setup.sh 6; else bash tools/env_setup.sh 2; fi

FROM base AS deploy
COPY --from=dev --chown=ubuntu:ubuntu /home/ubuntu/kana/thirdparty/intel-extension-for-pytorch/examples/cpu/inference/python/llm ./llm
RUN rm ./llm/tools/get_libstdcpp_lib.sh
COPY --from=dev --chown=ubuntu:ubuntu /home/ubuntu/kana/thirdparty/intel-extension-for-pytorch/examples/cpu/inference/python/llm/tools/get_libstdcpp_lib.sh ./llm/tools/get_libstdcpp_lib.sh
RUN . ./miniconda3/bin/activate && \
    conda create -y -n kana python=3.10 && conda activate kana && \
    echo "conda activate kana" >> ./.bashrc && \
    cd ./llm && \
    bash tools/env_setup.sh 1 && \
    python -m pip cache purge && \
    conda clean -a -y && \
    sudo mv ./oneCCL_release /opt/oneCCL && \
    sudo chown -R root:root /opt/oneCCL && \
    sed -i "s|ONECCL_PATH=.*|ONECCL_PATH=/opt/oneCCL|" ./tools/env_activate.sh