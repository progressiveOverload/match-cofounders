import requests
from bs4 import BeautifulSoup

# List of names
names = ['Alex Chiou', 'Rahul Pandey']

# Loop through names
for name in names:
    # Search for LinkedIn profile using Google search
    search_url = f'https://www.google.com/search?q={name} linkedin'
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get first search result
    search_result = soup.find('div', {'class': 'BNeawe UPmit AP7Wnd'})
    if search_result:
        # Extract LinkedIn profile URL
        linkedin_url = search_result.text
        if 'linkedin.com/in/' in linkedin_url:
            # Scrape skills from LinkedIn profile
            profile_url = linkedin_url.split('&')[0]  # Remove Google tracking params
            response = requests.get(profile_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            skills = soup.find_all('span', {'class': 'pv-skill-category-entity__name-text'})
            skills_list = [skill.text for skill in skills]

            # Print results
            print(f'{name}: {skills_list}')
        else:
            print(f'No LinkedIn profile found for {name}')
    else:
        print(f'No search results found for {name}')
