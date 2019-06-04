from linkedin import linkedin

APPLICATON_KEY    = '86xgm3qgahz8ah'
APPLICATON_SECRET = '9gOiZI4qOtsNyJ4T'

RETURN_URL = 'http://localhost:8888'

authentication = linkedin.LinkedInAuthentication(
                    APPLICATON_KEY,
                    APPLICATON_SECRET,
                    RETURN_URL)

print(authentication)
# Paste it here
# authentication.authorization_code = ''

result = authentication.get_access_token()

print ("Access Token:", result.access_token)
print ("Expires in (seconds):", result.expires_in)
