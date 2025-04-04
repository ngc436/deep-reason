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
    echo ""
    echo "Options for build-image:"
    echo "  --python-version VERSION  Override Python version (e.g., 3.10)"
}

function run_build_image {
    docker run -it -v $(pwd):/src poetry:py3.12 /bin/bash
}

function prepare_dist {
    echo "Exporting requirements..."
    poetry export --without-hashes --without-urls > requirements.txt
    
    echo "Building dist..."
    poetry build
    
    echo "Distribution files prepared successfully!"
}

function build_image {
    local python_version="3.12"
    local dockerfile_path="./docker/deep-reason.dockerfile"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --python-version)
                python_version="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    echo "Building Docker image: $IMAGE_NAME with Python $python_version"
    package_version=$(poetry version -s)
    
    # Create a temporary Dockerfile with the specified Python version
    temp_dockerfile=$(mktemp)
    sed "s/FROM python:.*/FROM python:${python_version}-bookworm/" "$dockerfile_path" > "$temp_dockerfile"
    
    docker build --build-arg "PACKAGE_VERSION=${package_version}" -t "$IMAGE_NAME" -f "$temp_dockerfile" .
    
    # Clean up temporary file
    rm "$temp_dockerfile"
    
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
	run-build-image)
	    run_build_image
	    ;;
        build-image)
            shift  # Remove the command name
            build_image "$@"
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
