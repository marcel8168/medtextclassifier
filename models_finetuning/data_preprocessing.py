import re
from xml.etree.ElementTree import ElementTree
import pandas as pd
from tqdm import tqdm

from global_parameters import LABELS_MAP


def xml_to_df(path: str, xml_file_names: list[str]):

    data_sets = [[], []]

    tree = ElementTree()

    hum_xml = tree.parse(path + xml_file_names[0])
    hum_records = hum_xml.findall('.//Rec')

    vet_xml = tree.parse(path + xml_file_names[1])
    vet_records = vet_xml.findall('.//Rec')

    record_sets = [hum_records, vet_records]

    progress_bar = tqdm(range(len(hum_records + vet_records)))

    for i, med_field in enumerate(LABELS_MAP):
        print(f"Processing medical field: {med_field}")
        labels = LABELS_MAP[med_field]
        for rec in record_sets[i]:
            try:
                common = rec.find('.//Common')
                pmid = common.find('PMID').text
                text_types = [elem.text for elem in common.findall('Type')]
                title = common.find('Title').text
                abstract = common.find('Abstract').text
                mesh_term_list = rec.find('.//MeshTermList')
                mesh_terms = [
                    term.text for term in mesh_term_list.findall('MeshTerm')]
            except Exception as e:
                print(f"An error occurred: {e}")
                print(f"Error occured for PMID: {pmid}")

            data_sets[i].append({'pmid': pmid, "text_types": text_types, 'title': preprocess_text(title),
                                 'abstract': preprocess_text(abstract), 'meshtermlist': mesh_terms, 'labels': labels})
            progress_bar.update(1)

    hum_df = pd.DataFrame(data_sets[0])
    vet_df = pd.DataFrame(data_sets[1])

    return hum_df, vet_df


def preprocess_text(text, lower_case=True, special_chars=True):
    text_after_case_processing = text.lower() if lower_case else text

    if special_chars:
        text_after_tab_processing = re.sub(
            r'[\r\n]+', ' ', text_after_case_processing)
        text_after_special_chars_processing = re.sub(
            r'[^\x00-\x7F]+', ' ', text_after_tab_processing)
    else:
        text_after_special_chars_processing = text_after_case_processing

    return text_after_special_chars_processing
