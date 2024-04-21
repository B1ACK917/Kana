image:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile -t kana:main .

naive-image:
    DOCKER_BUILDKIT=1 docker build -f Dockerfile_No_Acc -t kana-naive:main .

run:
	docker run --rm -it --privileged kana:main bash

run-naive:
    docker run --rm -it --privileged kana-naive:main bash
