#!/bin/bash
datahub docker quickstart --stop | true
datahub docker quickstart nuke | true
datahub docker quickstart
# datahub docker ingest-sample-data
datahub ingest -c ingest-config.yml