# TODO
FROM python:3.6-alpine3.6

COPY requirements.txt /src/
WORKDIR /src
RUN apk update \
    && apk add --no-cache make automake gcc g++ subversion python3-dev sqlite-dev ffmpeg
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
CMD ["sh", "start_prod.sh"]
