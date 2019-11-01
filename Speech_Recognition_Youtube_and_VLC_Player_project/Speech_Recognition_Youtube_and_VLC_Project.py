#!/usr/bin/env python
# coding: utf-8

# ## Speech Recognition using Python

# In[1]:


# Import Speech Recogition Package
import speech_recognition as spr


# In[34]:


# Validate the installation
spr.__version__


# In[11]:


recog = spr.Recognizer()


# In[33]:


recog.recognize_google()


# ### Convert Speech to Text

# In[39]:


speech = spr.AudioFile('speech.wav')
with speech as filesource:
    audio = recog.record(filesource)


# In[40]:


recog.recognize_google(audio)


# ### Convert Speech to Text - Capture only particular segments of audio using offset and duration

# In[41]:


with speech as filesource:
    audio = recog.record(filesource, duration=5)
    
recog.recognize_google(audio)


# In[42]:


# Capture multiple portions of speech one after another
with speech as filesource:
    audio_1 = recog.record(filesource, duration=5)
    audio_2 = recog.record(filesource, duration=5)

recog.recognize_google(audio_1)


# In[43]:


recog.recognize_google(audio_2)


# In[50]:


# Capturing second portion of the speech using an offset argument
with speech as filesource:
    audio = recog.record(filesource, offset=5, duration=7)

recog.recognize_google(audio)


# ### Convert Speech to Text - Effect of Noise

# In[51]:


noisyspeech = spr.AudioFile('noisy_speech.wav')

with noisyspeech as noisesource:
    audio = recog.record(noisesource)

recog.recognize_google(audio)


# In[52]:


with noisyspeech as noisesource:
    recog.adjust_for_ambient_noise(noisesource)
    audio = recog.record(noisesource)

recog.recognize_google(audio)


# In[27]:


recog.recognize_google(audio, show_all=True)


# ### Convert Speech to Text in Real Time using Microphone

# In[66]:


mc = spr.Microphone()


# In[67]:


#sr.Microphone.list_microphone_names()
mc.list_microphone_names()


# In[68]:


mc = spr.Microphone(device_index=0)


# In[33]:


with mc as source:
    audio = recog.listen(source)


# In[34]:


recog.recognize_google(audio)


# In[ ]:


#Reducing the effect of Noise
with mc as source:
    recog.adjust_for_ambient_noise(source)
    audio = recog.listen(source)


# ## Speech Recognition based Project

# In[88]:


#Import Necessary Libraries
import speech_recognition as spr
import webbrowser as wb
import pafy
import vlc
import urllib.request
from bs4 import BeautifulSoup
import time

#Create an empty list to store all the video URLs from the youtube.com page
linklist = []

#Create Recognizer() class objects called recog1 and recog2
recog1 = spr.Recognizer()
recog2 = spr.Recognizer()

#Create microphone instance with device microphone chosen whose index value is 0
mc = spr.Microphone(device_index=0)

#Capture voice
with mc as source:
    print("Search Youtube video to play")
    print("----------------------------")
    print("You can speak now")
    audio = recog1.listen(source)

#Based on speech, open youtube search page in a browser, get the first video link and play it in VLC media player
if 'search' in recog1.recognize_google(audio):
    recog1 = spr.Recognizer()
    url = 'https://www.youtube.com/results?search_query='
    with mc as source:
        print('Searching for the video(s)...')
        audio = recog2.listen(source)
        
        try:
            get_keyword = recog1.recognize_google(audio)
            print(get_keyword)
            wb.get().open_new(url+get_keyword)
            response = urllib.request.urlopen(url+get_keyword)
            html = response.read()
            soup = BeautifulSoup(html, 'html.parser')
            for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
                linklist.append('https://www.youtube.com' +vid['href'])
            videolink = pafy.new(linklist[1])
            bestlink = videolink.getbest()
            media = vlc.MediaPlayer(bestlink.url)
            media.play()
#             time.sleep(60)
#             media.stop()
        except spr.UnknownValueError:
            print("Unable to understand the input")
        except spr.RequestError as e:
            print("Unable to provide required output".format(e))


# In[89]:


media.stop()


# In[ ]:




