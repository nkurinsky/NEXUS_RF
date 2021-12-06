#! bin/bash

pandoc -f markdown -t html -s -o NexusReadmeView.html NexusKidReadme.md --metadata title="NEXUS KID Readme"
