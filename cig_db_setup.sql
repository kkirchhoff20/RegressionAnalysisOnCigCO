CREATE SCHEMA cig_db;
USE cig_db;
CREATE TABLE cig_data (
	id INT NOT NULL AUTO_INCREMENT,
    brand VARCHAR(50),
    tar DOUBLE,
    nicotine DOUBLE,
    mass DOUBLE,
    co DOUBLE,
    PRIMARY KEY (id)
    );
INSERT INTO cig_data VALUES
	(1, "Alpine", 14.1, 0.86, 0.9853, 13.6),
	(2, "Benson&Hedges", 16, 1.06, 1.0938, 16.6),
	(3, "BullDurham", 29.8, 2.03, 1.165, 23.5),
	(4, "CamelLights", 8, 0.67, 0.928, 10.2),
	(5, "Carlton", 4.1, 0.4, 0.9462, 5.4),
	(6, "Chesterfield", 15, 1.04, 0.8885, 15),
	(7, "GoldenLights", 8.8, 0.76, 1.0267, 9),
	(8, "Kent", 12.4, 0.95, 0.9225, 12.3),
	(9, "Kool", 16.6, 1.12, 0.9372, 16.3),
	(10, "L&M", 14.9, 1.02, 0.8858, 15.4),
	(11, "LarkLights", 13.7, 1.01, 0.9643, 13),
	(12, "Marlboro", 15.1, 0.9, 0.9316, 14.4),
	(13, "Merit", 7.8, 0.57, 0.9705, 10),
	(14, "MultiFilter", 11.4, 0.78, 1.124, 10.2),
	(15, "NewportLights", 9, 0.74, 0.8517, 9.5),
	(16, "Now", 1, 0.13, 0.7851, 1.5),
	(17, "OldGold", 17, 1.26, 0.9186, 18.5),
	(18, "PallMallLight", 12.8, 1.08, 1.0395, 12.6),
	(19, "Raleigh", 15.8, 0.96, 0.9573, 17.5),
	(20, "SalemUltra", 4.5, 0.42, 0.9106, 4.9),
	(21, "Tareyton", 14.5, 1.01, 1.007, 15.9),
	(22, "Trues", 7.3, 0.61, 0.9806, 8.5),
	(23, "ViceroyRichLight", 8.6, 0.69, 0.9693, 10.6),
	(24, "VirginiaSlims", 15.2, 1.02, 0.9496, 13.9),
	(25, "WinstonLights", 12, 0.82, 1.1184, 14.9);