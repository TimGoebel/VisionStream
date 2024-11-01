import json

def read_json_template(file_path):
    # Read the JSON file and extract the template string
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data['template']

def generate_report(job_function, comp_vision_item):
    # Read the template
    template = read_json_template('prompt.json')

    # Replace placeholders with actual values
    report = template.replace("[pretedetermine the job function here]", job_function)
    report = report.replace("[insert comp vision item]", comp_vision_item)

    return report