FROM python:3

WORKDIR /usr/src/astr

RUN ["apt-get", "update"]
RUN ["apt-get", "-y", "install", "vim", "tmux", "less"]

RUN ["python", "-m", "pip", "install", "scipy", "matplotlib", "ipython", "pandas", "sympy", "nose", "opencv-python", "opensimplex"]

CMD /bin/bash
