#!/bin/bash
wget -O "clip-vit-base-patch16_sae.pt" "https://drive.usercontent.google.com/download?id=1Ya00BYMezcMf6PEkRTf0m8e5lEUDEe0E&export=download&confirm=yes"
wget -O "clip-vit-base-patch16_sae-top32.pt" "https://drive.usercontent.google.com/download?id=1jezK-1zP8Ob0ADQ8A68agxYeCX_ekvlT&export=download&confirm=yes"
wget -O "clip-vit-base-patch16_sae-v2.pt" "https://drive.usercontent.google.com/download?id=1dedRPZCdw9ZPZS2u3QxC_bRRIrdE03oT&export=download&confirm=yes"
wget -O "clip-vit-base-patch16_sae-top32-v2.pt" "https://drive.usercontent.google.com/download?id=1kS1qtkoLNOQ8dxnAW5rGPQGAZ-0pL6Ai&export=download&confirm=yes"
wget -O "clip-vit-base-patch16_sae-v3.pt" "https://drive.usercontent.google.com/download?id=1zRPO4oqHUdc49rEq83dpF8sD39FWzsrv&export=download&confirm=yes"
wget -O "data.zip" "https://drive.usercontent.google.com/download?id=1Fp3xsPlJFJo2juPuZgrJdx8tp-_UhvmR&export=download&confirm=yes"
unzip data.zip
rm -f data.zip
