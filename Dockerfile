FROM python:3.9
# USER root
# RUN apt-get update
# RUN apt-get -y install locales && \
# localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
# ENV LANG ja_JP.UTF-8
# ENV LANGUAGE ja_JP:ja
# ENV LC_ALL ja_JP.UTF-8
# ENV TZ JST-9
# ENV TERM xterm
# COPY requirements.txt /root/
# RUN apt-get install -y vim less
# RUN pip install --upgrade pip
# RUN pip install --upgrade setuptools
# RUN pip install "cython<3.0.0" wheel
# # Install dm-launchpad from the wheel file using pip
# # RUN pip install /Users/sophie/GitRepositories/Project_Lab_Mizuho/gamma-vega-rl-hedging/dm_launchpad-0.5.0-cp39-cp39-manylinux2010_x86_64.whl
# RUN pip install "pyyaml==6.0" --no-build-isolation
# RUN pip install -r /root/requirements.txt
