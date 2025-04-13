from django import forms

class UploadPDFForm(forms.Form):
    file = forms.FileField()