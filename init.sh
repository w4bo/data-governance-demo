#!/usr/bin/env bash
set -exo

find . -type f -name "*.sh" -exec chmod +x {} \;

cp .env.example .env
