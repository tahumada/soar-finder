# soar-finder

Script to download finders for the SOAR telescope.
When optical surveys are requested it queries LS, PS1, DECaPs, DSS (in that order), and for IR surveys it querues 2MASS.

Stars are queried from Gaia, LS, PS1 (in that order, and 2MASS for the IR

usage:
python finder.py --ra XX  --dec XX*  --source-name XX

Replace XX with the values of RA, Dec, and the source name

* if Dec is negative please add '=' after --dec. For example use --dec=-11.3 instead of --dec -11.3

There are more options, as you can add a specific Position Angle (--pa-deg XX) or requesta finder in a different wavelenght regime (--wv ir , the default is --wv optical) 