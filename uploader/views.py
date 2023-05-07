from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from uploader.utils.data_preprocess import read_uploaded_file
from uploader.utils.summary import summarize, visualize

@csrf_protect
def index(request):
    if request.method == 'GET':
        return render(request, 'uploader.html', locals())
    
    if request.method == 'POST':
        uploaded_video = request.FILES['vid_files']      
        _, vid_path = read_uploaded_file(uploaded_video)
        summary_h5_path = summarize(vid_path)
        vid_path = visualize(summary_h5_path)
        
        return render(request, 'uploader.html', {'vid_path': vid_path})