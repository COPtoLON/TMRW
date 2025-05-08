
# Coinbase API
# username: mark@brezina.dk
# password: Ink-wire1x

from coinbase.rest import RESTClient
api_key = "organizations/3d3c116b-2b78-4e96-a5c5-671cc6d1857a/apiKeys/04dd8c95-f093-4485-a03d-4cb10dd88be2"
api_secret = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIEPljrHWwou2He+fBUmS79LPvL33TBV8mVciFp7s6kcAoAoGCCqGSM49\nAwEHoUQDQgAEZOqZjofgcT3cb90WM3BpckqkM8+tV1QOwTDNTC49QJQ39sxMhJu0\nOj3a/C81uosF+CQi2oqEij7OHGluo0DrMg==\n-----END EC PRIVATE KEY-----\n"
client = RESTClient(api_key=api_key, api_secret=api_secret)

# Kraken API
# username: mark@brezina.dk
# password: Ink-wire123x

from kraken.spot import SpotClient
key = "wwNo7XJL0OmK+iUpCUURdQakcohPuEwSYHMjLmqvtoKa2t2HG8qh4zIK"
secret = "murQoy7j9EUtGYDDwEY4KTl5f0nQdOer+bwdwm8Ybn2gsR44ol1nIV0cco5+xK5dcMfU4QCHmR5ea3LAE5ucVA=="
client.request("GET", "/0/public/Ticker?pair=XBTUSDC")



