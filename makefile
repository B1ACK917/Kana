image:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile -t kana:main .

naiveImage:
    DOCKER_BUILDKIT=1 docker build -f Dockerfile_No_Acc -t kanaNaive:main .

run:
	docker run --rm -it --privileged kana:main bash

runNaive:
    docker run --rm -it --privileged kanaNaive:main bash
