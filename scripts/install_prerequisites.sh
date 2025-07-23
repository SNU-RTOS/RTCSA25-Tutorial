#!/bin/bash

# install dev prerequisites
sudo apt install -y \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    build-essential \
    clang \
    cmake \
    unzip \
    pkg-config

# Install pyenv prerequisites
sudo apt install -y --no-install-recommends \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    ncurses-dev \
    libffi-dev \
    libreadline-dev \
    sqlite3 libsqlite3-dev \
    tk-dev \
    bzip2 libbz2-dev \
    lzma liblzma-dev \
    llvm libncursesw5-dev xz-utils libxml2-dev \
    libxmlsec1-dev 


# install Pyenv
curl -fsSL https://pyenv.run | bash

# update .bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init --path)"\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
source ~/.bashrc

# install python 3.10.16
pyenv install 3.10.16
pyenv global 3.10.16

# create python virtual environment
python3 -m venv .ws_pip
source .ws_pip/bin/activate
echo 'source $(pwd)/.ws_pip/bin/activate' >> ~/.zshrc

# install bazelisk and bazel
sudo curl -L https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-arm64 -o /usr/local/bin/bazelisk
sudo chmod +x /usr/local/bin/bazelisk
sudo ln -s /usr/local/bin/bazelisk /usr/bin/bazel
echo 'export HERMETIC_PYTHON_VERSION=3.10' >> ~/.zshrc

source ~/.bashrc


# (optional) install zsh
##################
# # Step 1: Install zsh, oh-my-zsh, powerlevel10k theme
# sudo apt install zsh -y
# sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
# git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

# # Step 2: Replace ZSH_THEME in ~/.zshrc
# sed -i 's/^ZSH_THEME=.*/ZSH_THEME="powerlevel10k\/powerlevel10k"/' ~/.zshrc

# # zsh-syntax-highlighting
# git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# # zsh-autosuggestions
# git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# # update .zshrc
# # 7. plugins 라인 수정 (기존에 있으면 교체)
# if grep -q '^plugins=' ~/.zshrc; then
#   sed -i 's/^plugins=.*/plugins=(git zsh-syntax-highlighting zsh-autosuggestions)/' ~/.zshrc
# else
#   echo 'plugins=(git zsh-syntax-highlighting zsh-autosuggestions)' >> ~/.zshrc
# fi

# # Step 3: Add p10k auto-config logic if not exists
# if ! grep -q 'p10k configure' ~/.zshrc; then
#   echo -e '\n[[ ! -f ~/.p10k.zsh ]] && p10k configure' >> ~/.zshrc
#   echo '[[ -f ~/.p10k.zsh ]] && source ~/.p10k.zsh' >> ~/.zshrc
# fi

# # Step 4: Set zsh as default shell (optional)
# chsh -s $(which zsh)

# # update .zshrc to support pyevn
# # echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
# # echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
# # echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init --path)"\n  eval "$(pyenv init -)"\nfi' >> ~/.zshrc
# # source ~/.zshrc
##################


# sudo apt install libopencv-dev -y
# wget -O opencv.zip https://github.com/opencv/opencv/archive/4.11.0.zip
# wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.11.0.zip
# unzip opencv.zip
# unzip opencv_contrib.zip
# mkdir opencv_build
# cd opencv_build
# cmake -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_INSTALL_PREFIX=/usr/src \
#     -DWITH_OPENCL=OFF \
#     -DWITH_OPENCL_SVM=OFF \
#     -DWITH_OPENCL_D3D11_NV=OFF \
#     -DWITH_OPENGL=OFF \
#     -DBUILD_opencv_world=ON \
#     -DBUILD_TESTS=OFF \
#     -DBUILD_PERF_TESTS=OFF \
#     -DWITH_TBB=OFF \
#     -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.11.0/modules/ \
#     ../opencv-4.11.0/

# make -j4
