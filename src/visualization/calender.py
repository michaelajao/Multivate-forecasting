from datetime import datetime, timedelta
import pandas as pd
import ics

# Define the start date for the schedule
start_date = datetime.now() + timedelta(days=1)  # Starting tomorrow
end_date = start_date + timedelta(weeks=4)  # Extending the schedule for 4 weeks

# Generate the daily schedule, taking into account the book chapter deadline
def generate_daily_schedule(date):
    if date.weekday() < 5:  # Weekdays
        return [
            {"start": date.replace(hour=7, minute=0), "end": date.replace(hour=7, minute=30), "title": "Morning Stretch/Walk"},
            {"start": date.replace(hour=7, minute=30), "end": date.replace(hour=8, minute=30), "title": "Breakfast and Personal Time"},
            {"start": date.replace(hour=8, minute=30), "end": date.replace(hour=11, minute=0), "title": "Coursera AI Study"},
            {"start": date.replace(hour=11, minute=0), "end": date.replace(hour=11, minute=30), "title": "Knee Recovery Exercises"},
            {"start": date.replace(hour=11, minute=30), "end": date.replace(hour=13, minute=0), "title": "Lunch and Rest"},
            {"start": date.replace(hour=13, minute=0), "end": date.replace(hour=15, minute=0), "title": "Book Chapter Writing"},
            {"start": date.replace(hour=15, minute=0), "end": date.replace(hour=15, minute=30), "title": "Break"},
            {"start": date.replace(hour=15, minute=30), "end": date.replace(hour=17, minute=30), "title": "PhD Research Work"},
            {"start": date.replace(hour=17, minute=30), "end": date.replace(hour=18, minute=0), "title": "Knee Exercises/Therapy"},
            {"start": date.replace(hour=18, minute=0), "end": date.replace(hour=19, minute=0), "title": "Dinner"},
            {"start": date.replace(hour=19, minute=0), "end": date.replace(hour=21, minute=0), "title": "Free Time"},
            {"start": date.replace(hour=21, minute=0), "end": date.replace(hour=22, minute=0), "title": "Wind Down"},
        ]
    elif date.weekday() == 5:  # Saturday
        return [
            {"start": date.replace(hour=9, minute=0), "end": date.replace(hour=11, minute=0), "title": "Recovery Exercises and Therapy"},
            {"start": date.replace(hour=11, minute=0), "end": date.replace(hour=23, minute=59), "title": "Leisure Activities"},
        ]
    else:  # Sunday
        return [
            {"start": date.replace(hour=0, minute=0), "end": date.replace(hour=23, minute=59), "title": "Day Off"},
        ]

# Generate the schedule for the month
schedule = []
current_date = start_date
while current_date < end_date:
    daily_schedule = generate_daily_schedule(current_date)
    for event in daily_schedule:
        schedule.append(event)
    current_date += timedelta(days=1)

# Create an .ics file from the schedule
calendar = ics.Calendar()
for event in schedule:
    if event['title'] != "Day Off" and event['title'] != "Leisure Activities":  # Add events to the calendar
        calendar_event = ics.Event(
            name=event['title'],
            begin=event['start'],
            end=event['end']
        )
        calendar.events.add(calendar_event)
    elif event['title'] == "Book Chapter Writing" and current_date > start_date + timedelta(days=14):
        # Stop adding book writing events after the deadline
        continue

# Save the .ics file
ics_file_path = '../../references/phd_student_schedule.ics'
with open(ics_file_path, 'w') as f:
    f.writelines(calendar)
ics_file_path
