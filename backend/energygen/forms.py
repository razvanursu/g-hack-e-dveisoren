from django import forms

class NameForm(forms.Form):
    latitude = forms.FloatField(required=True, label='Latitude')
    longitude = forms.FloatField(required=True, label='Longitude')
    time = forms.IntegerField(required=True, label='Time Difference')
    country = forms.CharField(required=True, label="Country", max_length=100)
