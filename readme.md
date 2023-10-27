# Faces

A light wrapper around [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition) to do facial recognition in Comfy.

## Shameless plug for my other nodes

Want to make your workflow cleaner? [UE Nodes](https://github.com/chrisgoringe/cg-use-everywhere) broadcast data without the wires.

Pause the workflow while you pick an image? [Image Picker](https://github.com/chrisgoringe/cg-image-picker).

## Install

You need to install dlib and face_recognition. This might work:

```
pip install dlib
pip install face_recognition
```

If it doesn't, go to [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition) and look at the instructions there.

Then install the custom node:

```
cd [comfy]/custom_nodes
git clone https://github.com/chrisgoringe/cg-faces
```

## Use

The nodes all take a single image of the face you are comparing with, and one or more candidate images. Each candidate is compared with the `true_image`, and given a score for the similarity of the face (just the face - nothing else is taken into account). An image which doesn't have exactly one face in scores 0, a perfect match scores 1. Above 0.7 is pretty good.

Nodes have a `message` output which is a string, either giving some similarity numbers or an error message. Install [display text from quicknodes](https://github.com/chrisgoringe/cg-quicknodes) if you don't have a node to display a text message.

Nodes either give a score, or sort the images, or return those above a threshold... they should be pretty obvious.

![example](docs/screenshot.png)