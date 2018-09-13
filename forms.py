from flask_wtf import FlaskForm, Form
from wtforms import StringField, BooleanField, SubmitField, DecimalField, SelectField
from wtforms.validators import DataRequired, Optional
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.fields.html5 import DecimalRangeField

from wtforms.validators import Required

def reset_features():
    features = {'rotate' : 0, 'grid' : 1, 'threshold' : 0.5 }
    return features

class RequiredIf2(DataRequired):
    """Validator which makes a field required if another field is set and has a truthy value.

    Sources:
        - http://wtforms.simplecodes.com/docs/1.0.1/validators.html
        - http://stackoverflow.com/questions/8463209/how-to-make-a-field-conditionally-optional-in-wtforms

    """
    field_flags = ('requiredif',)

    def __init__(self, other_field_name, message=None, *args, **kwargs):
        self.other_field_name = other_field_name
        self.message = message

    def __call__(self, form, field):
        other_field = form[self.other_field_name]
        if other_field is None:
            raise Exception('no field named "%s" in form' % self.other_field_name)
        if bool(other_field.data):
            super(RequiredIf, self).__call__(form, field)



class RequiredIf(Required):
    # a validator which makes a field required if   
    # another field is set and has a truthy value

    def __init__(self, other_field_name, *args, **kwargs):
        self.other_field_name = other_field_name
        super(RequiredIf, self).__init__(*args, **kwargs)

    def __call__(self, form, field):
        other_field = form._fields.get(self.other_field_name)
        if other_field is None:
            raise Exception('no field named "%s" in form' % self.other_field_name)
        if bool(other_field.data):
            super(RequiredIf, self).__call__(form, field)
            

            
class Upload(FlaskForm):    

    delete = SubmitField('Delete')  
    submit = SubmitField('Upload file')               
    veg = FileField('veg', validators=[RequiredIf('submit')])
    

class ImageUpdate(FlaskForm): 
            
    submit = SubmitField('Update')
    grid = DecimalField('grid', validators=[DataRequired()])
    threshold = DecimalField('threshold', validators=[DataRequired()])


class Storm(FlaskForm): 
            
    submit = SubmitField('Update duration')  
    
    tr = SelectField('Typical storm duration', 
            choices=[('0.5', '0.5'), ('1', '1'), ('3', '3'),('6', '6')])
    
                      
class Landscape(FlaskForm): 
            
    submit = SubmitField('Run random forest model')  
    reset = SubmitField('Reset form')  
    update = SubmitField('Update features')  
    p = SelectField('Typical storm intensity',  coerce=unicode, choices = [('', '')], validators=[Optional()])   
    slope = SelectField('slope',  choices=[('0.5', '0.5'), ('2', '2'), ('10', '10'), ('30', '30')])         
    # MAP = DecimalField('Mean annual precipitation', validators=[DataRequired()])        
    KsV = SelectField('Ksat', choices=[('2', '2'), ('4', '4'), ('8', '8'), ('16', '16')])         
      
# class UserDetails(Form):
#     group_id = SelectField(u'Group', coerce=int)

def edit_p(request, features):
    user = User.query.get(id)
    form = UserDetails(request.POST, obj=user)
    form.group_id.choices = [(g.id, g.name) for g in Group.query.order_by('name')] 
    
