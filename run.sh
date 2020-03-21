#!/bin/bash
echo "Starting container..."
docker start astr
docker attach astr
if [[ $? -eq 1 ]]; then
  echo "Building container..."
  docker build -t astr .
  echo "Starting container..."
  docker run -v $PWD:/usr/src/astr -t -i --name astr astr
fi
