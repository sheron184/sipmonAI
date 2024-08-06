import json
import csv

with open("cron_data.json", "r") as f:
    data = json.load(f)

# Step 2: Extract required properties and write to CSV
# Define the CSV field names
fieldnames = ['state', 'reason', 'event']


def train_hosts():
    # Step 3: Write to a CSV file
    with open('hosts.csv', 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write data rows
        for item in data["hosts"]:
            print(item["name"])
            try:
                output = item['output'].split('-')[1]
            except Exception as e:
                output = item['output'].split(':')[0]

            print('===============================================')
            if item['state'] == 0:
                event = 'host_ok'
            elif item['state'] == 1:
                event = 'host_down'
            elif item['state'] == 2:
                event = 'host_unreachable'
            else:
                event = 'unknown'
            writer.writerow({
                'state': item['state'],
                'reason': output,
                'event': event
            })


def train_services():
    with open('services.csv', 'w', newline='') as csv_file:
        fieldnames = ['state', 'reason', 'event', 'service']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write data rows
        for item in data["services"]:
            print(item["description"])
            try:
                output = item['output'].split('-')[1]
            except Exception as e:
                output = item['output'].split(':')[0]

            print('===============================================')
            if item['state'] == 0:
                event = 'service_ok'
            elif item['state'] == 1:
                event = 'service_critical'
            elif item['state'] == 2:
                event = 'service_warning'
            elif item['state'] == 3:
                event = 'service_unknown'
            else:
                event = 'unknown'
            writer.writerow({
                'state': item['state'],
                'reason': output,
                'event': event,
                'service': item['description']
            })
            
train_services()
