import csv

with open('data/item_meta.csv') as file:
    reader = csv.reader(file,delimiter=',')
    header = next(reader,None)
    data = [dict(zip(header,row)) for row in reader]
bought_together = [i['subtitle'] for i in data if i['subtitle']!= '']
print(bought_together)

numeric_cats = ['main']
a = {
    'item_id': '4110',
    'main_category': 'Tools & Home Improvement',
    'title': 'Scotch 6132-BA-10, 75-Inch x 66-Foot x 0.007-Inch, Pack of 10 Super 33+ Vinyl Electrical Tape, 10, Black, 10 Pack', 
    'average_rating': '4.8', 
    'rating_number': '514', 
    'features': "['INSULATES AND PROTECTS against abrasion and moisture', 'PROTECTIVE JACKETING up to 600V splice insulation', 'PRESSURE-SENSITIVE RUBBER-RESIN ADHESIVE and PVC backing for electrical and mechanical protection', 'INDOOR AND OUTDOOR USE', 'HIGH ADHESION in extreme temperatures: 0°F-221°F']", 
    'description': '["Scotch Delicate Surface Painter’s Tape is great for surfaces that require a little extra care such as wood floors, wallpaper, cabinets, painted drywall, and freshly painted walls* (*painted at least 24 hours ago). This delicate surface tape with gentle adhesive can stay on surfaces for up to 60 days and then removes easily without leaving any sticky residue behind. This tape features Edge-Lock Technology that seals out paint to deliver sharp paint lines and clean removal for your more sensitive projects, including accent walls, decorative stripes and patterns. Whether you\'re protecting your hardwood floors from paint splatter or going all out with a decorative mural, rely on Scotch Delicate Surface Painter\'s Tape for a professional look that\'s easy on your surfaces."]', 
    'price': '63.49', 
    'videos': "[{'title': 'Scotch Vinyl Electrical Tape (2086094)', 'url': 'https://www.amazon.com/vdp/087073f4671942b6ad6f56072d13ee4f?ref=dp_vse_rvc_0', 'user_id': ''}, {'title': '3M Super 33 Electrical Tape. 10PACK', 'url': 'https://www.amazon.com/vdp/0f300b66659d4987834c3b1cef1356c1?ref=dp_vse_rvc_1', 'user_id': '/shop/tacticalexpedition'}, {'title': '3M Built to perform (1937174)', 'url': 'https://www.amazon.com/vdp/05dcf01e755c4c20aee52d228ecd4a53?ref=dp_vse_rvc_2', 'user_id': ''}, {'title': 'Home Improvement Do’s and Don’ts', 'url': 'https://www.amazon.com/vdp/0071cd66693446a9b2657634fb48d161?ref=dp_vse_rvc_3', 'user_id': 'AGPOSSKWVV6YCIT76VCUYCHBDQ7Q'}, {'title': 'Scotch Vinyl Electrical Tape Super 88', 'url': 'https://www.amazon.com/vdp/4f307cadb0f94ff8b183eb0f0a76bfac?ref=dp_vse_rvc_4', 'user_id': ''}, {'title': 'Do you have cheap phone chargers that have exposed wires?', 'url': 'https://www.amazon.com/vdp/0e562fd2ce9e487186aabfdc8d5c279e?ref=dp_vse_rvc_5', 'user_id': '/shop/craigobligacionwilson'}, {'title': 'Not All Electrical Tape is Created Equally', 'url': 'https://www.amazon.com/vdp/74d226d9ec7746929f7593affd7fdcc1?ref=dp_vse_rvc_6', 'user_id': '/shop/metaspencer'}, {'title': 'Conductor Jacket Video', 'url': 'https://www.amazon.com/vdp/dd48df0ac83f4590b94f0128a9251ba5?ref=dp_vse_rvc_7', 'user_id': ''}, {'title': '3M Temflex Vinyl Electrical Tape 175', 'url': 'https://www.amazon.com/vdp/a8e3e6c64d5042c2a4890dfbbf613012?ref=dp_vse_rvc_8', 'user_id': ''}, {'title': '3M Safety SUPER-3/4X52FT black Super 33+ (TM) Electrical Tape, 3/4 in x 52 ft, 0. Fluid_Ounces', 'url': 'https://www.amazon.com/vdp/66aa3c6cbb87481fa29323734194ccd6?ref=dp_vse_rvc_9', 'user_id': ''}]", 
    'store': 'Scotch', 
    'categories': "['Industrial & Scientific', 'Adhesives, Sealants & Lubricants', 'Adhesive Tapes', 'Electrical Tape']", 
    'details': "{'Brand': 'Scotch', 'Color': 'Black', 'Material': 'Vinyl Plastic', 'Number of Items': '10', 'Special Feature': 'Rubber-resin', 'Surface Recommendation': 'Wood, Drywall', 'Size': '10', 'Thickness': '10 Mils', 'Tensile Strength': '15 Pounds Per Inch', 'Compatible Material': 'Vinyl, Wood, Polyvinyl Chloride', 'Manufacturer': '3M', 'Part Number': '6132-BA-10', 'Item Weight': '4.1 ounces', 'Item model number': '6132-BA-10', 'Is Discontinued By Manufacturer': 'No', 'Style': 'Tape', 'Item Package Quantity': '1', 'Special Features': 'Rubber-resin', 'Included Components': '10-Rolls', 'Batteries Included?': 'No', 'Batteries Required?': 'No', 'Best Sellers Rank': {'Industrial & Scientific': 14767, 'Electrical Tape': 73}, 'Date First Available': 'October 29, 2008'}", 
    'bought_together': '', # drop
    'subtitle': '', # drop
    'author': '' # drop
}