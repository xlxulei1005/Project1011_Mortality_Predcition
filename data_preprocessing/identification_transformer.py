def date_transformer(identification):
    
    def isdate(text):
        words = ['-','/','day','date','month','year','january','february','march','april',\
                'may','june','july','august','september','october','november','december']
        if any(x in text.lower() for x in words):
            return True
        else:
            return False

    if isdate(identification):
        return "DATE"
    else:
        return identification
    
def location_transformer(identification):
    
    def islocation(text):
        words = ['location','state','address','country']
        if any(x in text.lower() for x in words):
            return True
        else:
            return False
        
    if islocation(identification):
        return "LOCATION"
    else:
        return identification
        
def hospital_transformer(x):
    '''
    x: token
    return 'HOSPITAL'
    '''
    if 'hospital' in x.lower():
        return 'HOSPITAL'
    else:
        return x
    
def name_transformer(x):
    '''
    x:token 
    return 'NAME'
    '''
    if 'name' in x.lower():
        return 'NAME'
    else:
        return x
    
def company_transformer(x):
    '''
    x:token 
    return 'COMPANY'
    '''
    if 'company' in x.lower():
        return 'COMPANY'
    else:
        return x

def job_transformer(x):
    '''
    x:token 
    return 'JOB'
    '''
    if 'job' in x.lower():
        return 'JOB'
    else:
        return x



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def number_transformer(x):
    '''
    x:token 
    return 'NUMBER'
    '''
    if ('number' in x.lower()) or ('age' in x.lower()) or (is_number(x)):
        return 'NUMBER'
    else:
        return x 

def numerical_transformer(x):
    '''
    x:token 
    return 'NUMERIC_IDENTIFIER'
    '''
    if ('numerical' in x.lower()) or ('numeric' in x.lower()):
        return 'NUMERIC_IDENTIFIER'
    else:
        return x
    
def university_transformer(x):
    '''
    x:token 
    return 'UNIVERSITY'
    '''
    if ('university' in x.lower()):
        return 'UNIVERSITY'
    else:
        return x

def company_transformer(x):
    '''
    x:token 
    return 'NAME'
    '''
    if 'company' in x.lower():
        return 'COMPANY'
    else:
        return x

def contact_transformer(x):
    '''
    x:token 
    return 'CONTACT'
    '''
    if (('phone' in x.lower()) or ('contact' in x.lower())):
        return 'CONTACT'
    else:
        return x

def holiday_transformer(x):
    '''
    x:token 
    return 'HOLIDAY'
    '''
    if 'holiday' in x.lower():
        return 'HOLIDAY'
    else:
        return x
    
def url_transformer(x):
    '''
    x:token 
    return 'URL'
    '''
    if 'url' in x.lower():
        return 'URL'
    else:
        return x

def transformer(x):
    x = location_transformer(x)
    x = hospital_transformer(x)
    x = name_transformer(x)
    x = company_transformer(x)
    x = job_transformer(x)
    x = number_transformer(x)
    x = numerical_transformer(x)
    x = contact_transformer(x)
    x = holiday_transformer(x)
    x = url_transformer(x)
    x = university_transformer(x)
    x = date_transformer(x)
    return x
    
    