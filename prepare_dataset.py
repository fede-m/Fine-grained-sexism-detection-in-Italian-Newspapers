import glob
import json
from collections import defaultdict
import argparse
import random

"""
Input for newspaper: repubblica, corriere, lastampa

Input for category: tech, medicine, sport, economy, culture, school,
                    tv, science, world, cronaca, politics, regional, general, other, none

Input for perc_sample: float representing what percentage of the data you want to sample

"""


def parse_arguments():
    parser = argparse.ArgumentParser(description ="Specify which preprocessing actions you want to use")
    parser.add_argument("action", choices=["json_2_jsonl","txt_2_jsonl","create_distinct_datasets", "get_count_categories", "sample_per_category"], help="Specify the preprocessing action to perform")

    # Add action specific parameters
    parser.add_argument("--file_path", type= str, help="Specify a filepath for the files to be converted into json file")
    parser.add_argument("--out_filename", type= str, help = "Name of the output file")
    parser.add_argument("--jsonl_file", type=str, help="Name of the input json file")
    parser.add_argument("--newspaper", type=str, help = "Web site domain of the newspaper you want to filter")
    parser.add_argument("--category", type=str, help="What category should be filtered out?")
    parser.add_argument("--perc_sample", type= float, help="How many articles should be sampled?")

    return parser.parse_args()


webzio_file_path = r"../Datasets and web-scraping/642_20170904091358/642_webhose-2015-10-new_20170904091431/*.json"
cripco_file_path = r"../Datasets and web-scraping/CRIPCO_CORPUS/CRIPCO_CORPUS/CRIPCO_NePS_DEVSET/CRIPCO_NePS_devset_corpus/*/*.txt"

def json_2_jsonl(path_files, output_filename):
    # List all files in path

    article_files = glob.glob(path_files)
    articles =[]

    

    # Create list of dictionaries from articles files
    for article_file in article_files:
        with open(article_file, mode="r", encoding="utf-8") as article:
            #content = article.read()
            article_obj = json.load(article)
            articles.append(article_obj)

    # Create jsonl file from list of dictionaries
    write_json_file(output_filename, articles)


def txt_2_jsonl(path_files, output_filename):
    
    # List all filepaths
    article_files = glob.glob(path_files)
    articles = []

    for article in article_files:
        with open(article, mode="r", encoding="utf-8") as article:
            art = defaultdict()
            content = article.read()
            art["text"] = content.replace("\n"," ")
            articles.append(art)
    
    write_json_file(output_filename, articles)


def write_json_file(out_file, articles):
    with open(out_file, "w", encoding="utf-8") as outfile:
        for i,art in enumerate(articles):
            art = json.dumps(art, ensure_ascii=False)
            outfile.write(art +"\n")           


def create_distinct_datasets(jsonl_file, newspaper_site, output_file):
    articles = []
    with open(jsonl_file, mode= "r", encoding="utf-8") as jsonl_f:
        
            for line in jsonl_f:
                # Convert json string to dictionary
                json_object = json.loads(line)
                

                if newspaper_site in json_object["thread"]["site_full"]:
                    # art_object["title"] = json_object["title"]
                    # art_object["only_text"] = json_object["text"]
                    # art_object["site"] = json_object["thread"]["site_full"]
                    # art_object["text"] = json_object["title"] + " " + json_object["text"]
                    articles.append(json_object)
        
    write_json_file(output_file, articles)


def get_count_categories(jsonl_file, out_file_name):
    
    with open(jsonl_file, mode="r",encoding="utf-8") as json_f:
        categories = defaultdict(int)
        for line in json_f:
            json_obj = json.loads(line)
            category = json_obj["thread"]["section_title"]
            categories[category] += 1
    categories = [categories]

    write_json_file(out_file_name,categories)




def sample_per_category(json_file, newspaper, category, perc_sample, out_f):
    with open("categories.json", mode="r",encoding="utf-8") as category_file:
        categories = json.loads(category_file.read())[newspaper][category]
        
    articles = []
    with open(json_file, mode="r",encoding="utf-8") as json_f:
        for line in json_f:
            json_obj = json.loads(line)
            if (newspaper in json_obj["thread"]["site_full"] and json_obj["thread"]["section_title"] in categories):
                articles.append(json_obj)
    n_sample = round(len(articles)*perc_sample)
    articles = random.sample(articles,n_sample)
    write_json_file(out_f,articles)
        


def main():
    args = parse_arguments()

    if args.action == "json_2_jsonl":
        json_2_jsonl(args.file_path, args.out_filename)
    
    if args.action == "txt_2_jsonl":
        txt_2_jsonl(args.file_path, args.out_filename)
    
    if args.action == "create_distinct_datasets":
        create_distinct_datasets(args.jsonl_file, args.newspaper, args.out_filename)
    
    if args.action == "get_count_categories":
        get_count_categories(args.jsonl_file, args.out_filename)
    
    if args.action == "sample_per_category":
        sample_per_category(args.jsonl_file,args.newspaper, args.category, args.perc_sample, args.out_filename)


if __name__ == "__main__":
    main()