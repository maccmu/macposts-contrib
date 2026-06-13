# macposts-contrib

Various programs, scripts, and so on made with [macposts].

[macposts]: https://github.com/maccmu/macposts

Currently this repository contains two sub-projects:

- `dode`: Implementation of dynamic OD demand estimation models.
- `network_builder`: Scripts for reading, modifying, and writing data files
  which can be consumed by macposts.

If some of those scripts are proved to be useful, we may consider to add them to
the core macposts library.

## Getting started

These scripts target Python 3 and build on [macposts] plus several scientific
Python packages. Install the dependencies with:

    pip install -r requirements.txt

## Acknowledgment

This project is funded in part by Traffic 21 Institute, Carnegie Mellon
University's Mobility21, Technologies for Safe and Efficient Transportation
(T-SET), US Department of Transportation (DOT), and US Department of Energy (DOE).
The contents of this project reflect the views of the authors, who are
responsible for the facts and the accuracy of the information presented herein.
The US Government assumes no liability for the contents or use thereof.
