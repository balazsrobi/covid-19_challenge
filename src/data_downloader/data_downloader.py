import argparse
import kaggle
import os

"""
Make sure you downloaded the API from your account
and placed it ~/.kaggle folder.
You need to create that directory - the API does not do it for you. 
You must place the token there.

if you are using Linux.
1.pip install kaggle
2.cd ~/.kaggle
3.homepage www.kaggle.com -> Your Account -> Create New API token
4.mv ~/Downloads/kaggle.json ./
5.chmod 600 ./kaggle.json
"""

parser = argparse.ArgumentParser()


parser.add_argument('--folder', type=str,
                    help='Folder to save the downloaded data.')

parser.add_argument('--unzip', action='store_true',
                    help='Whether unzip the data or leave it untouched.')

args = parser.parse_args()

if not os.path.isdir(args.folder):

	try:
		os.mkdir(args.folder)
	except:
		print ("Creation of the directory %s failed" % args.folder)
	else:
		print ("Successfully created the directory %s" % args.folder)

kaggle.api.authenticate()

kaggle.api.dataset_download_files('allen-institute-for-ai/CORD-19-research-challenge', path=args.folder,
																					   unzip=args.unzip)