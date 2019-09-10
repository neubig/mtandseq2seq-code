# Attention code example
by Xinyi Wang

This is an example of a simple attention module, based on the pytorch examples.

## Data
	the file simply uses a list of dummy source encodings and one target encoding to illustrate how the code works. We provide both dot product attention module and Mlp attention module. 
## Basic Usage

	python attention.py

## Extra visualization
	We provide a visualization from an MT model using an open sourced git repo. You can try out the visualization by
	cd Attention-Visualization/
	python exec/plot_heatmap.py --input toydata/toy.attention

	Note that we slightly did some bug fix for the original repo code. If you are not using a mac, the Chinese characters may not show up on the plot correctly... 
