# Faces

A light wrapper around [serengil/deepface](https://github.com/serengil/deepface) to do facial recognition in Comfy.

## Shameless plug for my other nodes

Want to make your workflow cleaner? [UE Nodes](https://github.com/chrisgoringe/cg-use-everywhere) broadcast data without the wires.

Pause the workflow while you pick an image? [Image Picker](https://github.com/chrisgoringe/cg-image-picker).

## Install

You need to install deepface, and the default library used is facenet

```
pip install deepface
pip install facenet-pytorch
```

[Other possible dependencies](https://github.com/serengil/deepface/blob/master/requirements_additional.txt) - don't install these unless you're told to!

Then install the custom node:

```
cd [comfy]/custom_nodes
git clone https://github.com/chrisgoringe/cg-faces
```

Deepface uses various underlying models (check their page for details) which will download on first use.
 
## Use

The nodes all take a single image of the face you are comparing with, and one or more candidate images. Each candidate is compared with the `true_image`, and given a score for the similarity of the face (just the face - nothing else is taken into account). An image which doesn't have a face in scores 0 (if there are multiple faces in the candidate image the best match is used) a perfect match scores 1. Above 0.6 is generally considered a match.

Nodes have a `message` output which is a string, either giving some similarity numbers or an error message. Install [display text from quicknodes](https://github.com/chrisgoringe/cg-quicknodes) if you don't have a node to display a text message.

Nodes either give a score, or sort the images, or return those above a threshold... usage should be pretty obvious.

![example](docs/screenshot.png)