image:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile -t kana:main .

run:
	docker run --rm -it --privileged kana:main bash
