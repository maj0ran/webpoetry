FROM python:3.10
RUN pip install flask
RUN pip install uwsgi
RUN pip install numpy
RUN pip install tensorflow
RUN useradd -ms /bin/bash poem_user
USER poem_user
WORKDIR /app
COPY . /app
EXPOSE 5000
CMD uwsgi --http 0.0.0.0:5000 --module webpoet:poet --uid poem_user
