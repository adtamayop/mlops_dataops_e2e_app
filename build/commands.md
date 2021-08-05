
docker build --tag tfx_nuevo .

docker run \
    --name tfx_nuevo \
    --cpus="3.0" \
    --memory="6g" \
    --memory-reservation="3g" \
    -v $(pwd):/app \
    -d tfx_nuevo tail -f /dev/null

docker exec -it tfx_nuevo bash

docker rm -f tfx_nuevo