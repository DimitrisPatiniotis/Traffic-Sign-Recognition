# Traffic Sign Recognition

## Table of contents

* [Overview](#overview)
* [Goals](#goals)
* [Getting Started](#getting-started)
* [Run the experiments](#run-the-experiments)

## Overview

This repository includes the tools to create and modify CNNs that classify traffic signs. This task is being carried out as part of a study in CNNs and should be treated as such.

## Goals

The objective of this project is to create a modular component to create different CNNs and different datasets to train them.

## Getting Started

1. Clone repo

```
$ git clone git@github.com:DimitrisPatiniotis/Traffic-Sign-Recognition.git
```

2. Create a virtual environment and install all requirements listed in requirements.txt

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Run the experiments

To run the experiments run:

```
$ cd Processes/
$ python3 experiments.py
```

Note that you should first get the train and test data in tha Data/ folder.