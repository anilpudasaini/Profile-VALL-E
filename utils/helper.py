def decompose_style_id(style_id: torch.Tensor) -> Tuple[torch.Tensor]:
    """Map your style_ids to (age, gender, accent) indices"""
    mapping = {
        0: (0,0,0), 1: (1,0,0), 2: (2,0,0), 3: (0,1,0),
        4: (1,1,0), 5: (2,1,0), 6: (0,0,1), 7: (1,0,1),
        8: (2,0,1), 9: (0,1,1), 10: (1,1,1), 11: (2,1,1),
        12: (1,0,4), 13: (0,0,2), 14: (1,0,2), 15: (2,0,2),
        16: (0,1,2), 17: (1,1,2), 18: (2,1,2), 19: (0,0,3),
        20: (1,0,3), 21: (2,0,3), 22: (0,1,3), 23: (1,1,3),
        24: (2,1,3), 25: (0,0,4)
    }
    
    age_ids = torch.zeros_like(style_id)
    gender_ids = torch.zeros_like(style_id)
    accent_ids = torch.zeros_like(style_id)
    
    for b in range(style_id.size(0)):
        sid = style_id[b].item()
        age, gender, accent = mapping.get(sid, (0,0,0))
        age_ids[b] = age
        gender_ids[b] = gender
        accent_ids[b] = accent
        
    return age_ids, gender_ids, accent_ids