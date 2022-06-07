FROM python:3.10
RUN pip install flask
RUN pip install uwsgi
RUN useradd -ms /bin/bash poem_user
USER poem_user
WORKDIR /app
COPY gedichte_ki /app
#CMD ["uwsgi" "--http" "127.0.0.1:5000" "--module" "hello:app" "--uid" "poemwriter"]
EXPOSE 5000
CMD uwsgi --http 0.0.0.0:5000 --module hello:app --uid poem_user
#CMD uwsgi --socket 0.0.0.0:5000 --module hello:app
