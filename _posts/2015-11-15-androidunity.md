---
layout: post
title: Unity Android Application Integration
published: true
---

## **Unity Android Application Integration**

In this article we look at how to integrate an Unity and Android Project more specifically
how to add Android View and Layouts on top of exiting Unity Views.

### **Unity Application**

When a Unity app starts, `UnityPlayerProxyActivity` chooses which activity to instantiate: `UnityPlayerActivity` or `UnityPlayerNativeActivity`.

All three classes—UnityPlayerProxyActivity, UnityPlayerActivity, UnityPlayerNativeActivity—are compiled into a file called UnityPlayer.jar.

It chooses which class to instantiate based on the version of Android running on the device. Support for native activities was not added to Android until Gingerbread (API level 9) and so `UnityPlayerNativeActivity` is only instantiated if the device is running Gingerbread or higher.Otherwise, `UnityPlayerActivity` is instantiated.

Both UnityPlayerActivity and UnityPlayerNativeActivity are responsible for starting the Unity game engine
and therby instantiating the views and coroutines.

Let us consider the following Unity Program ,which just starts a coroutine upon initialization.
The scene contains no views or camera objects in the Unity Application just a `GameObject`

Let us add a script to the `gameobject` called `AndroidUnityInterface.cs`
The script is reponsible for running the routines on Unity as well as communicating with the android
Application.

![enter image description here](https://i.imgur.com/dmMxUL9.png)


```
using UnityEngine;
using System.Collections;

public class AndroidUnityInterface : MonoBehaviour {

	// Use this for initialization
	void Start () {
	
		Debug.LogError ("Starting the Android Unity Interface");
	}
	
	// Update is called once per frame
	void Update () {

		Debug.LogError ("Update the Android Unity Interface");
	
	}
}

```

This script will call the `Start` function while initialization and `Update` function is called at every frame.

### Unity Project Build
Build the project for the Android platform.Open the `Player settings` and take note of the package name that you have chosen as your "Bundle Identifier as `com.example.androidunityinterface` and the Android API level (e.g., 4.3 "JellyBean"), as you will reuse these settings later.

The same package name will be used while creating the android application in the below section.

If we compile and run the application,we would observe the Default Unity View and in the android Logs we would be able to observe the Debug Log statements written in the `Start` and `Update` functions.

```

11-15 16:46:20.276    7766-7791/? E/Unity? Starting the Android Unity Interface
    (Filename: ./artifacts/generated/common/runtime/UnityEngineDebug.gen.cpp Line: 56)
11-15 16:46:20.286    7766-7791/? E/Unity? Update the Android Unity Interface
    (Filename: ./artifacts/generated/common/runtime/UnityEngineDebug.gen.cpp Line: 56)
11-15 16:46:20.321    7766-7791/? E/Unity? Update the Android Unity Interface

```

### **Adding Android Views on Top of Unity**

With Unity Android it is possible to extend the standard `UnityPlayerActivity` or `UnityPlayerNativeActivity`  class (the primary Java class for the Unity Player on Android) enabling an application to override any and all of the basic interaction between Android OS and Unity Android.

Thus by extending the UnityPlayerNativeActivity we are manually instantiating the unity game engine.

#### Copying Unity Files

- As mentioned in the above section the `UnityPlayerNative` class is found in the classes.jar file in Unity. This files needs to be added to android project.

	The jar file can be found in 
`Editor\Data\PlaybackEngines\androidplayer\release\bin` path in Unity Installation directory.

	We need to add this `classes.jar` file to the project.

- we also need to copy unity libraries from following path from unity installation directory `Editor\Data\PlaybackEngines\androidplayer\development\libs\armeabi-v7a` to libs directory of the Android project.

Let first look at Android Application just instantiating the UnityGame Engine for the application in the above section.

We will be using `Android-Studio` for android application development

```

public class MainActivity extends UnityPlayerNativeActivity  {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        final long delay = 1000;//ms
        Handler handler = new Handler();

        Runnable runnable = new Runnable() {
            public void run() {
	            //gets the root default unity view
                ViewGroup rootView = (ViewGroup) MainActivity.this.findViewById(android.R.id.content);
                rootView.setKeepScreenOn(true);
            }
        }

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN | WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        
        handler.postDelayed(runnable, delay);
        
    }

}

```

### Native Events forwarding 
we need to allow native events forwarding so that events(like touch) are forwarded from unity to android.This is done by specifying the `ForwardNativeEventsToDalvik` setting in the manifest (must be set to true) 
```
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example.androidunityinterface" >

    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:theme="@style/AppTheme" >
        <activity
            android:name=".MainActivity"
            android:screenOrientation="landscape"
            android:label="@string/app_name"
            android:configChanges="fontScale|keyboard|keyboardHidden|locale|mnc|mcc|navigation|orientation|screenLayout|uiMode|touchscreen"
            >
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />

                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
            <meta-data
                android:name="unityplayer.ForwardNativeEventsToDalvik"
                android:value="true" />
        </activity>
    </application>

</manifest>

```

### Copying Unity Files

Use your file explorer to browse to your `Unity project directory`, where there is a folder called "/Temp" containing a subfolder called "/StagingArea." 

`StagingArea` folder contains these set of files and directories (such as /res, /assets, /bin,  AndroidManifest.xml,...) similar to a typical eclipse android directory.

Copy the contents of "/assets/" and "/libs" directories to corresponding directories in the AndroidProject.

### Testing The application

we can now run the android application and expect the same result as just running the unity application.

```

11-15 19:17:39.881      535-578/com.example.androidunityinterface E/Unity? Starting the Android Unity Interface
    UnityEngine.Debug:Internal_Log(Int32, String, Object)
    UnityEngine.Debug:LogError(Object)
    AndroidUnityInterface:Start()
    (Filename: ./artifacts/generated/common/runtime/UnityEngineDebug.gen.cpp Line: 56)
11-15 19:17:39.901      535-578/com.example.androidunityinterface E/Unity? Update the Android Unity Interface
    UnityEngine.Debug:Internal_Log(Int32, String, Object)
    UnityEngine.Debug:LogError(Object)
    AndroidUnityInterface:Update()
    (Filename: ./artifacts/generated/common/runtime/UnityEngineDebug.gen.cpp Line: 56)
11-15 19:17:39.966      535-578/com.example.androidunityinterface E/Unity? Update the Android Unity Interface
    UnityEngine.Debug:Internal_Log(Int32, String, Object)
    UnityEngine.Debug:LogError(Object)
    AndroidUnityInterface:Update()

```

### Adding  Android on Top of Unity Views

We can now add android views on top of the unity view programatically.

First  Scan the view hierarchy recursively, starting from the root view of your main activity, and find the leaf view in the hierarchy.

```
    private View getLeafView(View view) {
        if (view instanceof ViewGroup) {
            ViewGroup vg = (ViewGroup)view;
            for (int i = 0; i < vg.getChildCount(); ++i) {
                View chview = vg.getChildAt(i);
                View result = getLeafView(chview);
                if (result != null)
                    return result;
            }
            return null;
        }
        else {

            Log.e("ZZ","Found leaf view");
            return view;
        }
    }
```

When you find the leaf view, get the parent of that view. Call this view leafParent for ease of reference.

Add your custom views as a child of the leafParent view, for example, a layout inflated from XML.

```
public class MainActivity extends UnityPlayerNativeActivity  {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        final long delay = 1000;//ms
        Handler handler = new Handler();

        Runnable runnable = new Runnable() {
            public void run() {
                ViewGroup rootView = (ViewGroup) MainActivity.this.findViewById(android.R.id.content);
                rootView.setKeepScreenOn(true);

                //get the topmost view
                View topMostView = getLeafView(rootView);
                // let's add a sibling to the leaf view
                ViewGroup leafParent = (ViewGroup)topMostView.getParent();

                //inflate the android view to be added
                View view = getLayoutInflater().inflate(R.layout.activity_main, null, false);
                view.setKeepScreenOn(true);

                //add the android view on top of unity view
                leafParent.addView(view, new LayoutParams(LayoutParams.FILL_PARENT,LayoutParams.FILL_PARENT));
            }
        };

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN | WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        handler.postDelayed(runnable, delay);

    }
    
```

### Final Applicaiton

The "hello World" text is part of the android layout

![enter image description here](https://i.imgur.com/DSAVnfw.png)

The files for unity and android project can be found in the below `bitbucket` repository
https://bitbucket.org/pi19404/unityandroidproject/

**References**
- http://docs.unity3d.com/Manual/PluginsForAndroid.html
- https://developer.vuforia.com/library/articles/Solution/How-To-Add-Views-Over-Unity-for-Android