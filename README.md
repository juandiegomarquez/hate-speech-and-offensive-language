# Automated Hate Speech Detection and the Problem of Offensive Language
This repository is forked from [the original authors' repository](https://github.com/t-davidson/hate-speech-and-offensive-language): Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. 2017. "Automated Hate Speech Detection and the Problem of Offensive Language." ICWSM. You can read the paper [here](https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15665). The purpose of this fork is to port the pickled models to Python 3.


***WARNING: The data, lexicons, and notebooks all contain content that is racist, sexist, homophobic, and offensive in many other ways.***

Unchanged repositories:
* The `src` directory contains Python 2.7 and Python 3.6 code to replicate the authors' analyses in the paper. It has not been edited in any way in this repository
* The `lexicons` directory contains the lexicon they generated to try to more accurately classify hate speech

Changed/New Repositories:
* The `data` directory contains the labeled data as a csv. The pickle file containing the same data has been removed since it cannot be unpickled by Python 3.
* The new `pickled_models` directory contains the models necessary for classification. These are not the same pkl files from the original repository, as those were pickled with Python 2 and therefore cannot be unpickled with Python 3. The pkl files in this repository have all been pickled with Python 3.
* The `classifier` directory contains a script, instructions, and the necessary files to run the original authors' classifier on new data, and a test case is provided. To retrain the model and pickle it, simply re-run the train_models.py script.


***Please cite the original authors' paper in any published work that uses any of these resources.***
~~~
@inproceedings{hateoffensive,
  title = {Automated Hate Speech Detection and the Problem of Offensive Language},
  author = {Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar}, 
  booktitle = {Proceedings of the 11th International AAAI Conference on Web and Social Media},
  series = {ICWSM '17},
  year = {2017},
  location = {Montreal, Canada},
  pages = {512-515}
  }
~~~

***Contact Original Authors***
We would also appreciate it if you could fill out this short [form](https://docs.google.com/forms/d/e/1FAIpQLSdrPNlfVBlqxun2tivzAtsZaOoPC5YYMocn-xscCgeRakLXHg/viewform?usp=pp_url&entry.1506871634&entry.147453066&entry.1390333885&entry.516829772) if you are interested in using our data so we can keep track of how these data are used and get in contact with researchers working on similar problems.

If you have any questions please contact `trd54 at cornell dot edu`.
