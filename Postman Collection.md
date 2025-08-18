### Endpoints

GET 	| 	127.0.0.1:5001/ping

POST 	| 	127.0.0.1:5001/overlayFloor

POST 	| 	127.0.0.1:5001/overlayCarpet


### overlayFloor

1. JSON Body
```
{
  "room_image": "",
  "design_image": ""
}
```
2. JSON Response
```
{
  "final_output": "",
  "status": ""
}
```


### overlayCarpet
1. JSON Body
```
{
  "room_image": "",
  "carpet_image": "",
  "overlay_type": "ellipse(or trapezoid)",
  "carpet_dimensions": "8/6"
}
```
2. JSON Response
```
{
  "floor_mask_image": "",
  "status": "",
  "transparent_carpet_image": ""
}
```