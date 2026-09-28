from src.data_processing import classify_competition


def test_classify_competition_olympic():
    weight, label = classify_competition("Olympic Games Athens 2004")
    assert weight == 3
    assert label == "Jogos Olímpicos"

def test_classify_competition_world_championship():
    weight, label = classify_competition("World Championships Rio de Janeiro 2013")
    assert weight == 3
    assert label == "Campeonato Mundial"

def test_classify_competition_masters():
    weight, label = classify_competition("World Masters Qingdao 2019")
    assert weight == 3
    assert label == "World Masters"

def test_classify_competition_grand_slam():
    weight, label = classify_competition("Grand Slam Paris 2020")
    assert weight == 2
    assert label == "Grand Slam"

def test_classify_competition_grand_prix():
    weight, label = classify_competition("Grand Prix Zagreb 2019")
    assert weight == 2
    assert label == "Grand Prix"

def test_classify_competition_continental():
    weight, label = classify_competition("African Championships Seniors 2015")
    assert weight == 1
    assert label == "Campeonatos Continentais"

def test_classify_competition_copa_nacional():
    weight, label = classify_competition("European Open Madrid 2014")
    assert weight == 1
    assert label == "Copas Continentais e Nacionais"
    
    weight, label = classify_competition("World Cup Warsaw 2009")
    assert weight == 1
    assert label == "Copas Continentais e Nacionais"
