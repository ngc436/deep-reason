#!/usr/bin/env bash

set -e

IMAGE_NAME="node2.bdcl:5000/deep-reason:py3.12"

function show_usage {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  prepare-dist  Export requirements and build the distribution files"
    echo "  build-image   Build the Docker image"
    echo "  push         Push the built Docker image to the registry"
    echo "  help         Show this help message"
}

function prepare_dist {
    echo "Exporting requirements..."
    poetry export --without-hashes --without-urls > requirements.txt
    
    echo "Building dist..."
    poetry build
    
    echo "Distribution files prepared successfully!"
}

function build_image {
    echo "Building Docker image: $IMAGE_NAME"
    package_version=$(poetry version -s)
    docker build --build-arg "PACKAGE_VERSION=${package_version}" -t "$IMAGE_NAME" -f ./docker/deep-reason.dockerfile .
    
    echo "Build completed successfully!"
}

function push_image {
    echo "Pushing Docker image: $IMAGE_NAME"
    docker push "$IMAGE_NAME"
    echo "Push completed successfully!"
}

function main {
    # Main CLI logic
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi

    case "$1" in
        prepare-dist)
            prepare_dist
            ;;
        build-image)
            build_image
            ;;
        push)
            push_image
            ;;
        help)
            show_usage
            ;;
        *)
            echo "Error: Unknown command '$1'"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
