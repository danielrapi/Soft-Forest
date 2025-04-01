############################################ GOAL #####################################################
''' 
    This file will be dedicated to random helper function we think over time
'''
########################################################################################################

######################################### EMAIL ########################################################


import smtplib
from email.mime.text import MIMEText


def send_email(subject, body, to_email):
    '''
        Send a email to the user when the job is finished
    '''
    # sender_email = to_email
    sender_password = "hwxq ajsa wfyu xijy"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = to_email
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:  # Use your SMTP provider
        server.login(to_email, sender_password)
        server.sendmail(to_email, to_email, msg.as_string())


#############################################################################################################