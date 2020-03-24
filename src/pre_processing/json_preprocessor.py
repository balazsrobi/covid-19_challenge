# Source: https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv/output

import numpy as np
import pandas as pd
import json
import os
import tqdm
from copy import deepcopy


class JsonParser:

	def __init__(self, json_parent_folder, output_csv_folder):
		self.json_parent_folder = json_parent_folder
		self.output_csv_folder = output_csv_folder
		self.generate_clean_df()

	@staticmethod
	def format_name(author):
		middle_name = " ".join(author['middle'])
		
		if author['middle']:
			return " ".join([author['first'], middle_name, author['last']])
		else:
			return " ".join([author['first'], author['last']])

	@staticmethod        
	def format_affiliation(affiliation):
		text = []
		location = affiliation.get('location')
		if location:
			text.extend(list(affiliation['location'].values()))
		
		institution = affiliation.get('institution')
		if institution:
			text = [institution] + text
		return ", ".join(text)

	@staticmethod
	def format_authors(authors, with_affiliation=False):
		name_ls = []
		
		for author in authors:
			name = JsonParser.format_name(author)
			if with_affiliation:
				affiliation = JsonParser.format_affiliation(author['affiliation'])
				if affiliation:
					name_ls.append(f"{name} ({affiliation})")
				else:
					name_ls.append(name)
			else:
				name_ls.append(name)
		
		return ", ".join(name_ls)

	@staticmethod
	def format_body(body_text):
		texts = [(di['section'], di['text']) for di in body_text]
		texts_di = {di['section']: "" for di in body_text}
		
		for section, text in texts:
			texts_di[section] += text

		body = ""

		for section, text in texts_di.items():
			body += section
			body += "\n\n"
			body += text
			body += "\n\n"
		
		return body

	@staticmethod
	def format_bib(bibs):
		if type(bibs) == dict:
			bibs = list(bibs.values())
		bibs = deepcopy(bibs)
		formatted = []
		
		for bib in bibs:
			bib['authors'] = JsonParser.format_authors(
				bib['authors'], 
				with_affiliation=False
			)
			formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
			formatted.append(", ".join(formatted_ls))

		return "; ".join(formatted)


	def load_files(self):
		
		raw_files = []

		for root, dirs, files in os.walk(self.json_parent_folder):
			for name in files:

				if name.endswith("json"):
					file = json.load(open(os.path.join(root, name), 'rb'))
					raw_files.append(file)
		
		return raw_files

	def generate_clean_df(self):
		cleaned_files = []

		for root, dirs, files in os.walk(self.json_parent_folder):
			for name in files:

				if name.endswith("json"):

					file = json.load(open(os.path.join(root, name), 'rb'))

					features = [
								file['paper_id'],
								file['metadata']['title'],
								self.format_authors(file['metadata']['authors']),
								self.format_authors(file['metadata']['authors'], 
											   with_affiliation=True),
								self.format_body(file['abstract']),
								self.format_body(file['body_text']),
								self.format_bib(file['bib_entries']),
								file['metadata']['authors'],
								file['bib_entries']
								]
					cleaned_files.append(features)

		col_names = ['paper_id', 'title', 'authors',
					 'affiliations', 'abstract', 'text', 
					 'bibliography','raw_authors','raw_bibliography']

		clean_df = pd.DataFrame(cleaned_files, columns=col_names)
		
		clean_df.to_csv(os.path.join(self.output_csv_folder,'cleaned_output.csv'), index=False)


"""

Example of usage:
instance = JsonParser(json_parent_folder="/home/robertbalazs/projects/covid-19_challenge/data",
					  output_csv_folder="/home/robertbalazs/projects/covid-19_challenge/data")

"""